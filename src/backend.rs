use crate::parser::{
    BasicType, CompileError, Id, IdVec, Lexeme, Location, Node, NumericSpecification, Parser,
    Range, Token, TokenVec, Type,
};
use crate::semantic::Semantic;

use cranelift::codegen::ir::{
    condcodes::IntCC, FuncRef, MemFlags, Signature, StackSlotData, StackSlotKind,
};
use cranelift::prelude::{
    types, AbiParam, Ebb, ExternalName, FunctionBuilder, FunctionBuilderContext, InstBuilder,
    Value as CraneliftValue,
};
use cranelift_module::{default_libcall_names, DataContext, DataId, FuncId, Linkage, Module};
use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};

use std::collections::BTreeMap;

use std::sync::{Arc, Mutex};

use string_interner::Symbol;

type Sym = usize;

#[derive(Clone, Copy, Debug)]
pub struct FuncPtr(pub *const u8);

unsafe impl Send for FuncPtr {}
unsafe impl Sync for FuncPtr {}

const ENUM_TAG_SIZE_BYTES: u32 = 2;

// todo(chad): replace with evmap or similar?
// todo(chad): this doesn't have to be global if `dynamic_fn_ptr` function takes a `*Backend` parameter
lazy_static! {
    pub static ref FUNC_PTRS: Arc<Mutex<BTreeMap<Sym, FuncPtr>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Unassigned,
    None,
    FuncSym(Sym),
    FuncRef(FuncRef),
    Value(CraneliftValue),
    // todo(chad): not everything needs to a stack slot
    // using SSA Values where possible might be more efficient
    // StackSlot(StackSlot),
    // similar to a stackslot, but for general memory (or when the slot index isn't known, such as in a deref)
    // Address(CraneliftValue, u32),
}

impl Into<Value> for CraneliftValue {
    fn into(self) -> Value {
        Value::Value(self)
    }
}

impl Into<Value> for &CraneliftValue {
    fn into(self) -> Value {
        Value::Value(*self)
    }
}

impl Value {
    fn as_addr(&self) -> Value {
        *match self {
            Value::Value(_) => Some(self),
            _ => None,
        }
        .unwrap()
    }

    fn as_value_relaxed<'a, 'b, 'c, 'd>(
        &self,
        backend: &mut FunctionBackend<'a, 'b, 'c, 'd>,
    ) -> CraneliftValue {
        match self {
            Value::Value(v) => *v,
            Value::FuncSym(sym) => backend.builder.ins().iconst(types::I64, *sym as i64),
            _ => todo!("as_value_relaxed for {:?}", self),
        }
    }
}

pub struct Backend<'a, 'b> {
    pub semantic: &'a mut Semantic<'b>,
    pub module: Module<SimpleJITBackend>,
    pub string_literal_data_ids: BTreeMap<Sym, DataId>,
    pub values: Vec<Value>,
    pub funcs: BTreeMap<Sym, FuncId>,
    pub func_sigs: BTreeMap<Sym, Signature>,
    pub generation: u32,
    pub data_map: BTreeMap<Id, CraneliftValue>,
}

impl<'a, 'b> Backend<'a, 'b> {
    pub fn new(semantic: &'a mut Semantic<'b>) -> Self {
        let len = semantic.types.len();

        let mut module: Module<SimpleJITBackend> = {
            let mut jit_builder = SimpleJITBuilder::new(default_libcall_names());
            jit_builder.symbol("__dynamic_fn_ptr", dynamic_fn_ptr as *const u8);
            jit_builder.symbol("__push_token", push_token as *const u8);
            jit_builder.symbol("print_i8", print_i8 as *const u8);
            jit_builder.symbol("print_i16", print_i16 as *const u8);
            jit_builder.symbol("print_i32", print_i32 as *const u8);
            jit_builder.symbol("print_i64", print_i64 as *const u8);
            jit_builder.symbol("print_string", print_string as *const u8);
            jit_builder.symbol("alloc", alloc_helper as *const u8);

            Module::new(jit_builder)
        };

        let string_literal_data_ids = get_string_literal_data_ids(semantic, &mut module);

        Self {
            semantic,
            module,
            string_literal_data_ids,
            values: vec![Value::Unassigned; len],
            funcs: BTreeMap::new(),
            func_sigs: BTreeMap::new(),
            generation: 0,
            data_map: BTreeMap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn update_source_at_top_level(
        &mut self,
        source: &'b str,
        loc: Location,
    ) -> Result<(), CompileError> {
        self.generation += 1;

        self.semantic.parser.top_scope = 0;
        self.semantic.parser.function_scopes.clear();

        self.semantic
            .parser
            .lexer
            .update_source_for_copy(source, loc);

        self.semantic.parser.parse()?;

        self.semantic.topo.clear();
        self.semantic.types.clear();
        while self.semantic.types.len() < self.semantic.parser.nodes.len() {
            self.semantic.types.push(Type::Unassigned);
        }

        self.assign_top_level_types()?;

        Ok(())
    }

    pub fn generate_macro_call_order(
        &self,
        call_id: Id,
        started: &mut Vec<Id>,
        mac_topo: &mut Vec<Id>,
    ) {
        let macro_name = match self.semantic.parser.nodes[call_id] {
            Node::MacroCall { name, .. } => name,
            _ => unreachable!(),
        };
        let resolved_macro = self.semantic.resolve(macro_name).unwrap();
        let resolved_scope = self.semantic.parser.node_scopes[resolved_macro];

        if started.contains(&resolved_macro) {
            return;
        }

        started.push(resolved_macro);

        let calls_before = self
            .semantic
            .parser
            .macro_calls_by_function
            .get(&resolved_scope);

        if let Some(calls_before) = calls_before {
            for &cb in calls_before {
                if cb != call_id {
                    self.generate_macro_call_order(cb, started, mac_topo);
                }
            }
        }

        if !mac_topo.contains(&call_id) {
            mac_topo.push(call_id);
        }

        let pos = started.iter().position(|&e| e == resolved_macro).unwrap();
        started.swap_remove(pos);
    }

    pub fn assign_top_level_types(&mut self) -> Result<(), CompileError> {
        // ugh clone...
        for t in self.semantic.parser.externs.clone() {
            self.semantic.assign_type(t)?;
            self.compile_id(t)?;
        }

        let mut mac_topo = Vec::new();
        let mut mac_started = Vec::new();
        for t in self.semantic.parser.macro_calls.clone() {
            self.generate_macro_call_order(t, &mut mac_started, &mut mac_topo);
        }

        for t in mac_topo {
            self.semantic.assign_type(t)?;
            for t in self.semantic.topo.clone() {
                self.semantic.assign_type(t)?;
            }
            self.semantic.unify_types()?;
            self.allocate_for_new_nodes();

            for t in self.semantic.topo.clone() {
                self.compile_id(t)?;
            }
            self.semantic.topo.clear();
            self.compile_id(t)?;
        }

        // ugh clone...
        for tl in self.semantic.parser.top_level.clone() {
            let is_poly_or_macro = match &self.semantic.parser.nodes[tl] {
                Node::Func {
                    ct_params,
                    is_macro,
                    ..
                } => ct_params.is_some() || *is_macro,
                Node::Struct { ct_params, .. } => ct_params.is_some(),
                Node::Enum { ct_params, .. } => ct_params.is_some(),
                _ => false,
            };

            if !is_poly_or_macro {
                self.semantic.assign_type(tl)?
            }
        }

        self.semantic.unify_types()?;
        if !self.semantic.type_matches.is_empty() && !self.semantic.type_matches[0].is_empty() {
            CompileError::from_string(
                "Failed to unify types",
                self.semantic.parser.ranges[self.semantic.type_matches[0][0]],
            );
        }

        // for (index, ty) in self.types.iter().enumerate() {
        //     if !ty.is_concrete() {
        //         println!("Unassigned type for {}", self.parser.debug(index));
        //     }
        // }

        Ok(())
    }

    pub fn bootstrap_to_semantic(source: &str) -> Result<Semantic, CompileError> {
        FUNC_PTRS.lock().unwrap().clear();

        // let now = std::time::Instant::now();
        let mut parser = Parser::new(&source);
        parser.parse()?;
        // let elapsed = now.elapsed();
        // println!(
        //     "parser ran in: {}",
        //     elapsed.as_micros() as f64 / 1_000_000.0
        // );

        let semantic = Semantic::new(parser);

        Ok(semantic)
    }

    pub fn bootstrap_to_backend<'x, 'y>(
        semantic: &'x mut Semantic<'y>,
    ) -> Result<Backend<'x, 'y>, CompileError> {
        let mut backend = Backend::new(semantic);

        // let now = std::time::Instant::now();
        backend.assign_top_level_types()?;
        // let elapsed = now.elapsed();
        // println!(
        //     "semantic ran in: {}",
        //     elapsed.as_micros() as f64 / 1_000_000.0
        // );

        // let now = std::time::Instant::now();
        backend.compile()?;
        // let elapsed = now.elapsed();
        // println!(
        //     "backend ran in: {}",
        //     elapsed.as_micros() as f64 / 1_000_000.0
        // );

        Ok(backend)
    }

    #[allow(dead_code)]
    pub fn recompile_function(&mut self, function: impl Into<String>) -> Result<(), CompileError> {
        self.values.clear();
        while self.values.len() < self.semantic.parser.nodes.len() {
            self.values.push(Value::Unassigned);
        }

        // Replace all top-level function values with FuncRefs
        let new_id = self.semantic.parser.get_top_level(function).unwrap();
        for topo in self.semantic.topo.clone() {
            if topo == new_id {
                continue;
            }

            match self.semantic.parser.nodes[topo] {
                Node::Func { name, .. } => {
                    self.values[topo] = Value::FuncSym(name);
                }
                _ => {}
            };
        }

        self.compile_id(new_id)?;

        Ok(())
    }

    pub fn resolve_symbol(&self, sym: Sym) -> String {
        self.semantic
            .parser
            .lexer
            .string_interner
            .resolve(sym)
            .unwrap()
            .to_string()
    }

    pub fn get_symbol(&self, str: impl Into<String>) -> Sym {
        self.semantic
            .parser
            .lexer
            .string_interner
            .get(str.into())
            .unwrap()
    }

    pub fn call_func(&self, str: &str) -> i64 {
        // dbg!(FUNC_PTRS
        //     .lock()
        //     .unwrap()
        //     .keys()
        //     .map(|k| self.semantic.parser.lexer.resolve_unchecked(*k))
        //     .collect::<Vec<_>>());
        let f: fn() -> i64 = unsafe {
            std::mem::transmute(
                (FUNC_PTRS.lock().unwrap())
                    [&self.semantic.parser.lexer.string_interner.get(str).unwrap()],
            )
        };
        f()
    }

    pub fn compile(&mut self) -> Result<(), CompileError> {
        while self.values.len() < self.semantic.parser.nodes.len() {
            self.values.push(Value::Unassigned);
        }

        for tl in self.semantic.topo.clone() {
            self.compile_id(tl)?;
        }

        Ok(())
    }

    pub fn compile_id(&mut self, id: Id) -> Result<(), CompileError> {
        // idempotency
        match &self.values[id] {
            Value::Unassigned => {}
            _ => return Ok(()),
        };

        match self.semantic.parser.nodes[id] {
            Node::Func {
                name,      // Sym,
                params,    // IdVec,
                return_ty, // Id,
                stmts,     // IdVec,
                is_macro: _,
                copied_from: _,
                ..
            } => {
                let _now = std::time::Instant::now();

                let mut ctx = self.module.make_context();

                let func_name = String::from(self.semantic.parser.resolve_sym_unchecked(name));
                let func_sym = self.get_symbol(func_name.clone());

                self.values[id] = Value::FuncSym(func_sym);

                // println!("compiling func {}", func_name);
                // if let Some(copied_from) = copied_from {
                //     println!("compiling func {}, copied from {}", func_name, copied_from);
                // }

                let mut sig = self.module.make_signature();

                for param in self.semantic.parser.id_vec(params) {
                    sig.params
                        .push(AbiParam::new(into_cranelift_type(&self, *param)?));
                }

                let return_size = type_size(&self.semantic, &self.module, return_ty);
                if return_size > 0 {
                    sig.returns
                        .push(AbiParam::new(into_cranelift_type(&self, return_ty)?));
                }

                ctx.func.signature = sig;

                self.func_sigs.insert(func_sym, ctx.func.signature.clone());

                let func = self
                    .module
                    .declare_function(
                        &format!("{}_{}", func_name, self.generation),
                        Linkage::Local,
                        &ctx.func.signature,
                    )
                    .unwrap();
                ctx.func.name = ExternalName::user(0, func.as_u32());

                let mut func_ctx = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
                let ebb = builder.create_ebb();
                builder.switch_to_block(ebb);
                builder.append_ebb_params_for_function_params(ebb);

                let dfp_decl = {
                    let mut sig = self.module.make_signature();

                    sig.params.push(AbiParam::new(types::I64));
                    sig.returns
                        .push(AbiParam::new(self.module.isa().pointer_type()));

                    let func_id = self
                        .module
                        .declare_function("__dynamic_fn_ptr", Linkage::Import, &sig)
                        .unwrap();

                    self.module.declare_func_in_func(func_id, &mut builder.func)
                };

                let push_token_decl = {
                    let mut sig = self.module.make_signature();

                    sig.params
                        .push(AbiParam::new(self.module.isa().pointer_type())); // *Backend
                    sig.params.push(AbiParam::new(types::I64)); // unwrapped TokenVec
                    sig.params
                        .push(AbiParam::new(self.module.isa().pointer_type())); // *BackendToken

                    let func_id = self
                        .module
                        .declare_function("__push_token", Linkage::Import, &sig)
                        .unwrap();

                    self.module.declare_func_in_func(func_id, &mut builder.func)
                };

                declare_externs(self, &mut builder)?;

                let mut fb = FunctionBackend {
                    backend: self,
                    builder,
                    current_block: ebb,
                    dynamic_fn_ptr_decl: dfp_decl,
                    push_token_decl,
                };

                for stmt in fb.backend.semantic.parser.id_vec(stmts).clone() {
                    fb.compile_id(stmt)?;
                }

                fb.builder.seal_all_blocks();
                fb.builder.finalize();

                // println!("{}", ctx.func.display(None));

                self.module.define_function(func, &mut ctx).unwrap();
                self.module.clear_context(&mut ctx);
                self.data_map.clear();

                self.module.finalize_definitions();

                let func = self.module.get_finalized_function(func);
                FUNC_PTRS.lock().unwrap().insert(func_sym, FuncPtr(func));

                // let elapsed = now.elapsed().as_micros();
                // if copied_from.is_some() {
                //     println!("compiled copied fn in {} millis", elapsed as f64 / 1000.0);
                // }

                Ok(())
            }
            Node::Symbol(sym) => {
                let resolved = self.semantic.scope_get(sym, id)?;
                self.compile_id(resolved)?;
                self.semantic.parser.node_is_addressable[id] =
                    self.semantic.parser.node_is_addressable[resolved];
                self.values[id] = self.values[resolved];

                Ok(())
            }
            Node::MacroCall {
                name,
                input,
                expanded: _,
            } => {
                self.values[id] = Value::None;

                self.compile_id(name)?;

                // let input = self.semantic.parser.token_vec(input).clone();
                let name = match self.semantic.parser.nodes[name] {
                    Node::Symbol(sym) => sym,
                    _ => unreachable!(),
                };

                // let compiled_macro = self.build_hoisted_function(id, params, name)?;
                // let compiled_macro: fn(semantic: *mut Semantic) -> i64 = unsafe { std::mem::transmute(compiled_macro) };
                let compiled_macro: fn(*const BackendTokenArray) -> *mut BackendTokenArray =
                    unsafe { std::mem::transmute((FUNC_PTRS.lock().unwrap())[&name]) };

                let mut backend_tokens = self.to_backend_token_array(input);
                let mut backend_token_array = BackendTokenArray {
                    length: backend_tokens.len() as _,
                    data: backend_tokens.as_mut_ptr(),
                };

                let result_tokens = compiled_macro(&mut backend_token_array as *mut _);
                let result_tokens: Vec<BackendToken> = unsafe {
                    Vec::from_raw_parts(
                        (*result_tokens).data,
                        (*result_tokens).length as _,
                        (*result_tokens).length as _,
                    )
                };

                let tokens_id = self.semantic.parser.push_token_vec(Vec::new());
                for rt in result_tokens.iter() {
                    push_token(self.semantic as *mut _, tokens_id.0 as _, rt as *const _)
                }
                std::mem::forget(result_tokens);

                let tokens = self.semantic.parser.token_vecs[tokens_id.0 as usize].clone();
                self.semantic.parser.lexer.macro_tokens = Some(tokens);
                self.semantic.parser.top_scope = self.semantic.parser.node_scopes[id];
                self.semantic.parser.copying = false; // Macro-generated code doesn't count as copying
                self.semantic.parser.lexer.top = Lexeme::new(Token::Eof, Range::default());
                self.semantic.parser.lexer.second = Lexeme::new(Token::Eof, Range::default());
                self.semantic.parser.lexer.pop();
                self.semantic.parser.lexer.pop();

                // println!(
                //     "received tokens: {:?}",
                //     self.semantic.parser.token_vecs[tokens_id as usize].clone()
                // );

                // todo(chad): infer what to parse by where the macro is being instantiated (fn stmt, top level, expression, etc.)
                // let parsed = self.semantic.parser.parse_fn_stmt()?;
                let parsed = self.semantic.parser.parse_top_level()?;

                self.semantic.allocate_for_new_nodes();
                self.allocate_for_new_nodes();

                self.semantic.assign_type(parsed)?;

                self.semantic.parser.lexer.macro_tokens = None;

                Ok(())
            }

            Node::TypeLiteral(_) => {
                self.values[id] = Value::None;
                Ok(())
            }
            Node::Extern { .. } => Ok(()),
            _ => Err(CompileError::from_string(
                format!("Unhandled node: {}", self.semantic.parser.debug(id)),
                self.semantic.parser.ranges[id],
            )),
        }
    }

    fn to_backend_token_array(&self, tv: TokenVec) -> Vec<BackendToken> {
        let mut tokens = Vec::new();
        let mut strings = Vec::new();
        for t in self.semantic.parser.token_vecs.get(tv.0).unwrap() {
            tokens.push(self.to_backend_token(t, &mut strings));
        }
        std::mem::forget(strings);

        return tokens;
    }

    fn to_backend_token(&self, lex: &Lexeme, strings: &mut Vec<String>) -> BackendToken {
        let tag = match lex.tok {
            Token::LParen => 0,
            Token::RParen => 1,
            Token::LCurly => 2,
            Token::RCurly => 3,
            Token::LSquare => 4,
            Token::RSquare => 5,
            Token::DoubleLAngle => 6,
            Token::DoubleRAngle => 7,
            Token::LAngle => 8,
            Token::RAngle => 9,
            Token::Return => 10,
            Token::Semicolon => 11,
            Token::Colon => 12,
            Token::Bang => 13,
            Token::Dot => 14,
            Token::Asterisk => 15,
            Token::EqEq => 16,
            Token::Neq => 17,
            Token::Add => 18,
            Token::Sub => 19,
            Token::Mul => 20,
            Token::Div => 21,
            Token::Eq => 22,
            Token::Comma => 23,
            Token::Struct => 24,
            Token::Enum => 25,
            Token::Fn => 26,
            Token::Macro => 27,
            Token::Extern => 28,
            Token::TypeOf => 29,
            Token::Cast => 30,
            Token::If => 31,
            Token::While => 32,
            Token::Else => 33,
            Token::True => 34,
            Token::False => 35,
            Token::I8 => 36,
            Token::I16 => 37,
            Token::I32 => 38,
            Token::I64 => 39,
            Token::Ampersand => 40,
            Token::Type => 41,
            Token::Caret => 42,
            Token::Uninit => 43,
            Token::Symbol(_) => 44,
            Token::IntegerLiteral(_, _) => 45,
            Token::FloatLiteral(_) => 46,
            Token::StringLiteral(_) => 47,
            _ => unreachable!(),
        };

        let data = match lex.tok {
            Token::Symbol(sym) | Token::StringLiteral(sym) => {
                let sym_str: String = self.resolve_symbol(sym);
                let len = unsafe { std::mem::transmute::<usize, [u8; 8]>(sym_str.len()) };

                let data = sym_str.as_ptr();
                let data = unsafe { std::mem::transmute::<_, [u8; 8]>(data) };

                strings.push(sym_str);

                unsafe { std::mem::transmute::<_, [u8; 16]>([len, data]) }
            }
            Token::FloatLiteral(f) => unsafe {
                let f = std::mem::transmute::<f64, [u8; 8]>(f);
                std::mem::transmute::<_, [u8; 16]>([f, [0u8; 8]])
            },
            Token::IntegerLiteral(n, _) => unsafe {
                let n = std::mem::transmute::<i64, [u8; 8]>(n);
                std::mem::transmute::<_, [u8; 16]>([n, [0u8; 8]])
            },
            _ => [0u8; 16],
        };

        BackendToken { tag, data }
    }

    pub fn allocate_for_new_nodes(&mut self) {
        self.semantic.allocate_for_new_nodes();
        while self.values.len() < self.semantic.parser.nodes.len() {
            self.values.push(Value::Unassigned);
        }
    }

    // Returns whether the value for a particular id is the value itself, or a pointer to the value
    // For instance, the constant integer 3 would likely be just the value, and not a pointer.
    // However a field access (or even a load) needs to always be returned by pointer, because it might refer to a struct literal.
    // Some nodes (like const int above) don't usually return a pointer, but they might need to for other reasons,
    // e.g. they need their address to be taken. In that case we store that fact in the 'node_is_addressable' vec.
    // Also symbols won't know whether they have a local or not because it depends what they're referencing. So 'node_is_addressable' is handy there too.
    fn rvalue_is_ptr(&self, id: Id) -> bool {
        if self.semantic.parser.node_is_addressable[id] {
            return true;
        }

        match self.semantic.parser.nodes[id] {
            Node::Load(load_id) => self.rvalue_is_ptr(load_id),
            Node::ValueParam { value, .. } => self.rvalue_is_ptr(value),
            Node::Field { .. } => true,
            Node::Let { .. } => true,
            Node::DeclParam {
                ct_link: Some(ct_link),
                ..
            } => self.rvalue_is_ptr(ct_link),
            _ => false,
        }
    }
}

fn get_string_literal_data_ids(
    semantic: &Semantic,
    module: &mut Module<SimpleJITBackend>,
) -> BTreeMap<Id, DataId> {
    let mut string_literal_data_ids = BTreeMap::new();

    for lit in semantic.parser.string_literals.clone() {
        // declare data section for string literals
        let string_literal_data_id = module
            .declare_data(&format!("str_lit_{}", lit), Linkage::Local, false, None)
            .unwrap();

        let mut data_ctx = DataContext::new();
        let mut lits = String::new();
        let lit_sym = match semantic.parser.nodes[lit] {
            Node::StringLiteral { sym, .. } => sym,
            _ => unreachable!(),
        };
        let lit_str = semantic
            .parser
            .lexer
            .string_interner
            .resolve(lit_sym)
            .unwrap()
            .to_string();

        lits += lit_str.as_str();

        let raw_bytes: Box<[u8]> = lits.into_boxed_str().into_boxed_bytes();
        data_ctx.define(raw_bytes);

        module
            .define_data(string_literal_data_id, &data_ctx)
            .unwrap();

        string_literal_data_ids.insert(lit_sym, string_literal_data_id);
    }

    string_literal_data_ids
}

pub struct FunctionBackend<'a, 'b, 'c, 'd> {
    pub backend: &'a mut Backend<'b, 'c>,
    pub builder: FunctionBuilder<'d>,
    pub current_block: Ebb,
    pub dynamic_fn_ptr_decl: FuncRef,
    pub push_token_decl: FuncRef,
}

impl<'a, 'b, 'c, 'd> FunctionBackend<'a, 'b, 'c, 'd> {
    fn set_value(&mut self, id: Id, value: CraneliftValue) {
        self.backend.values[id] = Value::Value(value);
    }

    fn as_value(&mut self, id: Id) -> CraneliftValue {
        match self.backend.values[id] {
            Value::Value(v) => Some(v),
            Value::FuncSym(sym) => Some(self.builder.ins().iconst(types::I64, sym as i64)),
            _ => None,
        }
        .unwrap_or_else(|| {
            panic!(
                "Cannot convert {:?} into a CraneliftValue",
                &self.backend.values[id]
            )
        })
    }

    fn store(&mut self, id: Id, dest: &Value) {
        if self.backend.rvalue_is_ptr(id) {
            self.store_copy(id, dest);
        } else {
            self.store_value(id, dest, None);
        }
    }

    fn store_with_offset(&mut self, id: Id, dest: &Value, offset: i32) {
        if self.backend.rvalue_is_ptr(id) {
            let dest = if offset == 0 {
                *dest
            } else {
                let dest = dest.as_value_relaxed(self);
                let dest = self.builder.ins().iadd_imm(dest, offset as i64);
                Value::Value(dest)
            };

            self.store_copy(id, &dest);
        } else {
            self.store_value(id, dest, Some(offset));
        }
    }

    fn store_value(&mut self, id: Id, dest: &Value, offset: Option<i32>) {
        let source_value = self.backend.values[id].clone().as_value_relaxed(self);

        match dest {
            Value::Value(value) => {
                self.builder.ins().store(
                    MemFlags::new(),
                    source_value,
                    *value,
                    offset.unwrap_or_default(),
                );
            }
            _ => todo!("store_value dest for {:?}", dest),
        }
    }

    fn store_copy(&mut self, id: Id, dest: &Value) {
        let size = type_size(&self.backend.semantic, &self.backend.module, id);

        let source_value = self.as_value(id);
        let dest_value = dest.as_value_relaxed(self);

        self.builder.emit_small_memcpy(
            self.backend.module.isa().frontend_config(),
            dest_value,
            source_value,
            size as _,
            1,
            1,
        );
    }

    fn rvalue(&mut self, id: Id) -> CraneliftValue {
        if self.backend.rvalue_is_ptr(id) {
            let ty = &self.backend.semantic.types[id];
            let ty = match ty {
                Type::Struct { .. } => self.backend.module.isa().pointer_type(),
                Type::Enum { .. } => self.backend.module.isa().pointer_type(),
                _ => into_cranelift_type(&self.backend, id).unwrap(),
            };

            match self.backend.values[id] {
                Value::Value(v) => self.load(ty, v, 0),
                _ => panic!(
                    "Could not get cranelift value: {:?}",
                    &self.backend.values[id]
                ),
            }
        } else {
            self.as_value(id)
        }
    }

    pub fn compile_id(&mut self, id: Id) -> Result<(), CompileError> {
        // idempotency
        match &self.backend.values[id] {
            Value::Unassigned => {}
            _ => return Ok(()),
        };

        match self.backend.semantic.parser.nodes[id] {
            Node::Return(return_id) => {
                self.compile_id(return_id)?;

                let value = self.rvalue(return_id);
                self.backend.values[id] = Value::Value(value);

                let return_size =
                    type_size(&self.backend.semantic, &self.backend.module, return_id);
                if return_size > 0 {
                    self.builder.ins().return_(&[value]);
                } else {
                    self.builder.ins().return_(&[]);
                }

                Ok(())
            }
            Node::IntLiteral(i, _) => {
                let value = match self.backend.semantic.types[id] {
                    Type::Basic(bt) => match bt {
                        BasicType::I64 => self.builder.ins().iconst(types::I64, i),
                        BasicType::I32 => self.builder.ins().iconst(types::I32, i),
                        BasicType::I16 => self.builder.ins().iconst(types::I16, i),
                        BasicType::I8 => self.builder.ins().iconst(types::I8, i),
                        _ => todo!(
                            "handle other type: {:?} {:?}",
                            &self.backend.semantic.types[id],
                            &self.backend.semantic.parser.nodes[id]
                        ),
                    },
                    _ => todo!("expected basic type"),
                };

                self.set_value(id, value);

                if self.backend.semantic.parser.node_is_addressable[id] {
                    let size = type_size(&self.backend.semantic, &self.backend.module, id);
                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                        offset: None,
                    });
                    let slot_addr = self.builder.ins().stack_addr(
                        self.backend.module.isa().pointer_type(),
                        slot,
                        0,
                    );
                    let value = Value::Value(slot_addr);
                    self.store_value(id, &value, None);
                    self.backend.values[id] = value;
                }

                Ok(())
            }
            Node::BoolLiteral(b) => {
                let value = self.builder.ins().iconst(types::I32, if b { 1 } else { 0 });
                self.set_value(id, value);

                Ok(())
            }
            Node::StructLiteral { name: _, params } => {
                // if this typechecked as an enum, handle things a bit differently
                if let Type::Enum { .. } = self.backend.semantic.types[id] {
                    let params = self.backend.semantic.parser.id_vec(params);

                    assert_eq!(params.len(), 1);
                    let param = params[0];
                    let size = type_size(&self.backend.semantic, &self.backend.module, param)
                        + ENUM_TAG_SIZE_BYTES;

                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                        offset: None,
                    });
                    let addr = self.builder.ins().stack_addr(
                        self.backend.module.isa().pointer_type(),
                        slot,
                        0,
                    );

                    self.backend.values[id] = Value::Value(addr);

                    let index = match &self.backend.semantic.parser.nodes[param] {
                        Node::ValueParam { index, .. } => index,
                        _ => unreachable!(),
                    };
                    let tag_value = self.builder.ins().iconst(types::I16, *index as i64);
                    self.builder
                        .ins()
                        .store(MemFlags::new(), tag_value, addr, 0);

                    let tag_size_value = self
                        .builder
                        .ins()
                        .iconst(types::I64, ENUM_TAG_SIZE_BYTES as i64);
                    let addr = self.builder.ins().iadd(addr, tag_size_value);

                    self.compile_id(param)?;
                    self.store(param, &addr.into());

                    return Ok(());
                }

                // todo(chad): @Optimization: this is about the slowest thing we could ever do, but works great.
                // Come back later once everything is working and make it fast
                let size = type_size(&self.backend.semantic, &self.backend.module, id);
                let slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let addr = self.builder.ins().stack_addr(
                    self.backend.module.isa().pointer_type(),
                    slot,
                    0,
                );

                self.backend.values[id] = Value::Value(addr);

                let mut offset: i32 = 0;
                for param in self.backend.semantic.parser.id_vec(params).clone() {
                    self.compile_id(param)?;
                    self.store_with_offset(param, &Value::Value(addr), offset);
                    offset += type_size(&self.backend.semantic, &self.backend.module, param) as i32;
                }

                Ok(())
            }
            Node::Symbol(sym) => {
                let resolved = self.backend.semantic.scope_get(sym, id)?;
                self.compile_id(resolved)?;
                self.backend.semantic.parser.node_is_addressable[id] =
                    self.backend.semantic.parser.node_is_addressable[resolved];
                self.backend.values[id] = self.backend.values[resolved];

                Ok(())
            }
            Node::EqEq(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                // todo(chad): @Optimization utilize the `br_icmp` instruction in cranelift for the `if a == b` or `if a != b` case
                let value = self.builder.ins().icmp(IntCC::Equal, lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::Neq(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                // todo(chad): @Optimization utilize the `br_icmp` instruction in cranelift for the `if a == b` or `if a != b` case
                let value = self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::LessThan(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                // todo(chad): @Optimization utilize the `br_icmp` instruction in cranelift for the `if a == b` or `if a != b` case
                let value = self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::GreaterThan(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                // todo(chad): @Optimization utilize the `br_icmp` instruction in cranelift for the `if a == b` or `if a != b` case
                let value = self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::Add(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                let value = self.builder.ins().iadd(lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::Sub(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                let value = self.builder.ins().isub(lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::Mul(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                let value = self.builder.ins().imul(lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::Div(lhs, rhs) => {
                self.compile_id(lhs)?;
                self.compile_id(rhs)?;

                let lhs = self.rvalue(lhs);
                let rhs = self.rvalue(rhs);

                let value = self.builder.ins().sdiv(lhs, rhs);
                self.set_value(id, value);

                // todo(chad): store if necessary
                Ok(())
            }
            Node::Let {
                name: _, // Sym
                ty: _,   // Id
                expr,    // Id
            } => {
                let size = type_size(&self.backend.semantic, &self.backend.module, id);
                let slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });

                let slot_addr = self.builder.ins().stack_addr(
                    self.backend.module.isa().pointer_type(),
                    slot,
                    0,
                );
                let value = Value::Value(slot_addr);

                if let Some(expr) = expr {
                    self.compile_id(expr)?;
                    self.store(expr, &value);
                }

                self.backend.values[id] = value;

                Ok(())
            }
            Node::Set {
                name,     // Id
                expr,     // Id
                is_store, // bool
            } => {
                if is_store {
                    // Don't actually codegen the load, just codegen what it is loading so we have the pointer. Then it's easy to store into
                    let name = match self.backend.semantic.parser.nodes[name] {
                        Node::Load(load_id) => load_id,
                        _ => unreachable!(),
                    };

                    self.compile_id(name)?;
                    self.compile_id(expr)?;

                    let mut addr = self.backend.values[name].clone().as_value_relaxed(self);
                    if self.backend.rvalue_is_ptr(name) {
                        addr = self.load(self.backend.module.isa().pointer_type(), addr, 0);
                    }

                    self.store(expr, &Value::Value(addr));
                } else {
                    self.compile_id(name)?;

                    let addr = self.backend.values[name].as_addr();

                    self.compile_id(expr)?;
                    self.store(expr, &addr);
                }

                Ok(())
            }
            Node::MacroCall { expanded, .. } => {
                for stmt in self.backend.semantic.parser.id_vec(expanded).clone() {
                    self.compile_id(stmt)?;
                }

                Ok(())
            }
            Node::Call {
                name,   // Id
                params, // IdVec
                ..
            } => {
                let params = self.backend.semantic.parser.id_vec(params).clone();
                self.compile_call(id, name, &params)
            }
            Node::If {
                cond,
                true_stmts,
                false_stmts,
            } => {
                self.compile_id(cond)?;
                let cond = self.rvalue(cond);

                let true_block = self.builder.create_ebb();
                self.builder
                    .append_ebb_params_for_function_params(true_block);

                let false_block = self.builder.create_ebb();
                self.builder
                    .append_ebb_params_for_function_params(false_block);

                let cont_block = self.builder.create_ebb();
                self.builder
                    .append_ebb_params_for_function_params(cont_block);

                let current_params = self.builder.ebb_params(self.current_block).to_owned();

                self.builder.ins().brnz(cond, true_block, &current_params);
                self.builder.ins().jump(false_block, &current_params);

                // true
                {
                    self.builder.switch_to_block(true_block);
                    self.current_block = true_block;

                    for stmt in self.backend.semantic.parser.id_vec(true_stmts).clone() {
                        self.compile_id(stmt)?;
                    }

                    let current_params = self.builder.ebb_params(self.current_block).to_owned();
                    self.builder.ins().jump(cont_block, &current_params);
                }

                // false
                {
                    self.builder.switch_to_block(false_block);
                    self.current_block = false_block;

                    for stmt in self.backend.semantic.parser.id_vec(false_stmts).clone() {
                        self.compile_id(stmt)?;
                    }

                    let current_params = self.builder.ebb_params(self.current_block).to_owned();
                    self.builder.ins().jump(cont_block, &current_params);
                }

                self.builder.switch_to_block(cont_block);
                self.current_block = cont_block;

                Ok(())
            }
            Node::While { cond, stmts } => {
                let check_block = self.builder.create_ebb();
                self.builder
                    .append_ebb_params_for_function_params(check_block);

                let true_block = self.builder.create_ebb();
                self.builder
                    .append_ebb_params_for_function_params(true_block);

                let cont_block = self.builder.create_ebb();
                self.builder
                    .append_ebb_params_for_function_params(cont_block);

                // put the check and branch inside a block
                let current_params = self.builder.ebb_params(self.current_block).to_owned();
                self.builder.ins().jump(check_block, &current_params);
                self.builder.switch_to_block(check_block);
                self.current_block = check_block;

                self.compile_id(cond)?;
                let cond = self.rvalue(cond);

                let current_params = self.builder.ebb_params(self.current_block).to_owned();
                self.builder.ins().brnz(cond, true_block, &current_params);
                self.builder.ins().jump(cont_block, &current_params);

                // true
                {
                    self.builder.switch_to_block(true_block);
                    self.current_block = true_block;

                    for stmt in self.backend.semantic.parser.id_vec(stmts).clone() {
                        self.compile_id(stmt)?;
                    }

                    let current_params = self.builder.ebb_params(self.current_block).to_owned();
                    self.builder.ins().jump(check_block, &current_params);
                }

                self.builder.switch_to_block(cont_block);
                self.current_block = cont_block;

                Ok(())
            }
            Node::Ref(ref_id) => {
                self.compile_id(ref_id)?;
                match self.backend.values[ref_id] {
                    Value::Value(_) => self.backend.values[id] = self.backend.values[ref_id],
                    _ => todo!("{:?}", &self.backend.values[ref_id]),
                };

                // if we are meant to be addressable, then we need to store our actual value into something which has an address
                if self.backend.semantic.parser.node_is_addressable[id] {
                    // todo(chad): is there any benefit to creating all of these up front?
                    let size = self.backend.module.isa().pointer_bytes() as u32;
                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                        offset: None,
                    });
                    let slot_addr = self.builder.ins().stack_addr(
                        self.backend.module.isa().pointer_type(),
                        slot,
                        0,
                    );
                    let value = Value::Value(slot_addr);
                    self.store_value(id, &value, None);
                    self.backend.values[id] = value;
                }

                Ok(())
            }
            Node::Load(load_id) => {
                self.compile_id(load_id)?;

                let value = match self.backend.values[load_id] {
                    Value::Value(value) => value,
                    _ => todo!("load for {:?}", &self.backend.values[load_id]),
                };

                let ty = &self.backend.semantic.types[id];
                let ty = match ty {
                    Type::Struct { .. } => self.backend.module.isa().pointer_type(),
                    Type::Enum { .. } => self.backend.module.isa().pointer_type(),
                    Type::Array(_) => self.backend.module.isa().pointer_type(),
                    _ => into_cranelift_type(&self.backend, id)?,
                };

                let loaded = if self.backend.rvalue_is_ptr(load_id) {
                    self.load(types::I64, value, 0)
                } else {
                    self.load(ty, value, 0)
                };

                self.set_value(id, loaded);

                Ok(())
            }
            Node::DeclParam {
                name: _,  // Sym
                ty: _,    // Id
                index,    // u16
                is_ct: _, // bool
                ct_link,  // Option<Id>
            } => {
                if let Some(ct_link) = ct_link {
                    self.compile_id(ct_link)?;
                    self.backend.values[id] = self.backend.values[ct_link];

                    self.backend.semantic.parser.node_is_addressable[id] =
                        self.backend.semantic.parser.node_is_addressable[ct_link];

                    Ok(())
                } else {
                    // we need our own storage
                    let size = type_size(&self.backend.semantic, &self.backend.module, id);
                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                        offset: None,
                    });

                    let slot_addr = self.builder.ins().stack_addr(
                        self.backend.module.isa().pointer_type(),
                        slot,
                        0,
                    );
                    let value = Value::Value(slot_addr);

                    let params = self.builder.ebb_params(self.current_block);
                    let param_value = params[index as usize];
                    self.builder
                        .ins()
                        .store(MemFlags::new(), param_value, slot_addr, 0);

                    self.backend.values[id] = value;

                    Ok(())
                }
            }
            Node::ValueParam {
                name: _, // Sym
                value,
                ..
            } => {
                self.compile_id(value)?;
                self.backend.values[id] = self.backend.values[value];
                Ok(())
            }
            Node::Field {
                base,
                field_name,
                field_index,
                is_assignment,
            } => {
                self.compile_id(base)?;

                // todo(chad): deref more than once?
                let (unpointered_ty, loaded) = match self.backend.semantic.types[base] {
                    Type::Pointer(id) => (id, true),
                    _ => (base, false),
                };

                // field access on a string
                let is_array_like_ty = match self.backend.semantic.types[unpointered_ty] {
                    Type::String | Type::Array(_) => true,
                    _ => false,
                };

                if is_array_like_ty {
                    let field_name = String::from(
                        self.backend
                            .semantic
                            .parser
                            .resolve_sym_unchecked(field_name),
                    );

                    let mut base = self.as_value(base);

                    if loaded {
                        // if we are doing field access through a pointer, do an extra load
                        base = self.load(self.backend.module.isa().pointer_type(), base, 0);
                    }

                    match field_name.as_str() {
                        "len" => (),
                        "ptr" => {
                            base = self.builder.ins().iadd_imm(base, 8);
                        }
                        _ => unreachable!(),
                    }

                    self.backend.values[id] = base.into();
                    return Ok(());
                }

                // all field accesses on enums have the same offset -- just the tag size in bytes
                let is_enum = match &self.backend.semantic.types[unpointered_ty] {
                    Type::Enum { .. } => true,
                    _ => false,
                };

                let offset = if is_enum {
                    ENUM_TAG_SIZE_BYTES
                } else {
                    let params = self.backend.semantic.types[unpointered_ty]
                        .as_struct_params()
                        .unwrap();

                    self.backend
                        .semantic
                        .parser
                        .id_vec(params)
                        .iter()
                        .enumerate()
                        .map(
                            |(_index, id)| match self.backend.semantic.parser.nodes[*id] {
                                Node::DeclParam { ty, .. } => ty,
                                Node::ValueParam { value, .. } => value,
                                _ => unreachable!(),
                            },
                        )
                        .take(field_index as usize)
                        .map(|ty| type_size(&self.backend.semantic, &self.backend.module, ty))
                        .sum::<u32>()
                };

                let mut base = self.as_value(base);

                if loaded {
                    // if we are doing field access through a pointer, do an extra load
                    base = self.load(self.backend.module.isa().pointer_type(), base, 0);
                }

                if is_enum && is_assignment {
                    let tag_value = self.builder.ins().iconst(types::I16, field_index as i64);
                    self.builder
                        .ins()
                        .store(MemFlags::new(), tag_value, base, 0);
                } else {
                    // todo(chad): @Correctness check the tag, assert it is correct
                }

                if offset != 0 {
                    let offset = self.builder.ins().iconst(types::I64, offset as i64);
                    base = self.builder.ins().iadd(base, offset);
                }

                self.backend.values[id] = base.into();

                Ok(())
            }
            Node::ArrayLiteral(elements) => {
                let ty = match self.backend.semantic.types[id] {
                    Type::Array(ty) => ty,
                    _ => unreachable!(),
                };

                let element_count = self.backend.semantic.parser.id_vec(elements).len();

                let len_value = self.builder.ins().iconst(types::I64, element_count as i64);

                let struct_addr_value = {
                    let element_size = type_size(&self.backend.semantic, &self.backend.module, ty);
                    let struct_size = type_size(&self.backend.semantic, &self.backend.module, ty)
                        * element_count as u32;
                    let struct_slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: struct_size,
                        offset: None,
                    });
                    let struct_slot_addr = self.builder.ins().stack_addr(
                        self.backend.module.isa().pointer_type(),
                        struct_slot,
                        0,
                    );
                    let struct_addr_value = Value::Value(struct_slot_addr);

                    let mut offset = 0;
                    for value in self.backend.semantic.parser.id_vec(elements).clone() {
                        self.compile_id(value)?;
                        self.store_with_offset(value, &struct_addr_value, offset);
                        offset += element_size as i32;
                    }

                    struct_slot_addr
                };

                let size = type_size(&self.backend.semantic, &self.backend.module, id);
                let struct_slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let dest_addr = self.builder.ins().stack_addr(
                    self.backend.module.isa().pointer_type(),
                    struct_slot,
                    0,
                );

                // store the length
                self.builder
                    .ins()
                    .store(MemFlags::new(), len_value, dest_addr, 0);

                // store the ptr
                self.builder
                    .ins()
                    .store(MemFlags::new(), struct_addr_value, dest_addr, 8);

                self.backend.values[id] = Value::Value(dest_addr);

                Ok(())
            }
            Node::StringLiteral { sym, bytes } => {
                let lit_id = self.backend.string_literal_data_ids[&sym];

                let string_lit_ptr = self
                    .backend
                    .module
                    .declare_data_in_func(lit_id, &mut self.builder.func);
                let global = self
                    .builder
                    .ins()
                    .global_value(self.backend.module.isa().pointer_type(), string_lit_ptr);

                let len_value = self.builder.ins().iconst(types::I64, bytes as i64);

                let size = type_size(&self.backend.semantic, &self.backend.module, id);
                let struct_slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let dest_addr = self.builder.ins().stack_addr(
                    self.backend.module.isa().pointer_type(),
                    struct_slot,
                    0,
                );

                // store the length
                self.builder
                    .ins()
                    .store(MemFlags::new(), len_value, dest_addr, 0);

                // store the ptr
                self.builder
                    .ins()
                    .store(MemFlags::new(), global, dest_addr, 8);

                self.backend.values[id] = Value::Value(dest_addr);

                Ok(())
            }
            Node::ArrayAccess { arr, index } => {
                self.compile_id(arr)?;
                self.compile_id(index)?;

                let struct_ptr = self.backend.values[arr].as_addr().as_value_relaxed(self);
                let data_ptr = self.load(self.backend.module.isa().pointer_type(), struct_ptr, 8);

                let element_size = type_size(&self.backend.semantic, &self.backend.module, id);
                let element_size = self.builder.ins().iconst(types::I64, element_size as i64);

                let index = self.rvalue(index);
                let offset = self.builder.ins().imul(element_size, index);

                self.backend.values[id] = Value::Value(self.builder.ins().iadd(data_ptr, offset));

                Ok(())
            }
            Node::Cast(expr) => {
                self.compile_id(expr)?;
                self.backend.values[id] = self.backend.values[expr];
                Ok(())
            }
            Node::TypeOf(expr) => {
                let value = self.get_global_ptr_to_type_of(expr);
                self.set_value(id, value);

                Ok(())
            }
            Node::TypeLiteral(_) => {
                self.backend.values[id] = Value::None;
                Ok(())
            }
            _ => Err(CompileError::from_string(
                format!(
                    "Unhandled node: {:?} ({})",
                    &self.backend.semantic.parser.nodes[id],
                    self.backend.semantic.parser.debug(id)
                ),
                self.backend.semantic.parser.ranges[id],
            )),
        }
    }

    fn declare_global_data_with_size(&mut self, name: &str, bytes: u32) -> CraneliftValue {
        let data_id = self
            .backend
            .module
            .declare_data(name, Linkage::Local, true, None)
            .unwrap();

        let mut data_ctx = DataContext::new();
        data_ctx.define_zeroinit(bytes as usize);

        self.backend.module.define_data(data_id, &data_ctx);

        let ptr = self
            .backend
            .module
            .declare_data_in_func(data_id, &mut self.builder.func);

        self.builder
            .ins()
            .symbol_value(self.backend.module.isa().pointer_type(), ptr)
    }

    fn declare_global_data_with_bytes(&mut self, name: &str, bytes: Box<[u8]>) -> CraneliftValue {
        let data_id = self
            .backend
            .module
            .declare_data(name, Linkage::Local, true, None)
            .unwrap();

        let mut data_ctx = DataContext::new();
        data_ctx.define(bytes);

        self.backend.module.define_data(data_id, &data_ctx).unwrap();

        self.backend.module.finalize_definitions();
        self.backend.module.get_finalized_data(data_id);

        let ptr = self
            .backend
            .module
            .declare_data_in_func(data_id, &mut self.builder.func);

        self.builder
            .ins()
            .symbol_value(self.backend.module.isa().pointer_type(), ptr)
    }

    fn get_global_ptr_to_type_of(&mut self, id: Id) -> CraneliftValue {
        if let Some(value) = self.backend.data_map.get(&id) {
            return *value;
        }

        let ty = self.backend.semantic.types[id];

        // allocate struct
        let struct_size = type_size(
            &self.backend.semantic,
            &self.backend.module,
            self.backend.semantic.parser.ty_decl.unwrap(),
        );

        let tag = match ty {
            Type::Basic(BasicType::I8) => 0,
            Type::Basic(BasicType::I16) => 1,
            Type::Basic(BasicType::I32) => 2,
            Type::Basic(BasicType::I64) => 3,
            Type::Basic(BasicType::Bool) => 4,
            Type::Basic(BasicType::F32) => 5,
            Type::Basic(BasicType::F64) => 6,
            Type::String => 7,
            Type::Pointer(_) => 8,
            Type::Array(_) => 9,
            Type::Struct { .. } => 10,
            Type::Enum { .. } => 11,
            Type::Func { .. } => 12,
            _ => todo!("support #type_of for other types: {:?}", ty),
        };
        let dest_addr = self.declare_global_data_with_size(&format!("{}_typeof", id), struct_size);
        self.backend.data_map.insert(id, dest_addr);

        // store the tag
        let tag = self.builder.ins().iconst(types::I16, tag);
        self.builder.ins().store(MemFlags::new(), tag, dest_addr, 0);

        // store the data
        match ty {
            Type::Pointer(pointer_id) => {
                let pointer_ty_ptr = self.get_global_ptr_to_type_of(pointer_id);
                self.builder.ins().store(
                    MemFlags::new(),
                    pointer_ty_ptr,
                    dest_addr,
                    ENUM_TAG_SIZE_BYTES as i32,
                );
            }
            Type::Array(array_id) => {
                let array_ty_ptr = self.get_global_ptr_to_type_of(array_id);
                self.builder.ins().store(
                    MemFlags::new(),
                    array_ty_ptr,
                    dest_addr,
                    ENUM_TAG_SIZE_BYTES as i32,
                );
            }
            Type::Struct { params, .. } => {
                self.fill_type_of_for_struct_like_data(id, params, dest_addr)
            }
            Type::Enum { params, .. } => {
                self.fill_type_of_for_struct_like_data(id, params, dest_addr)
            }
            Type::Func { .. } => todo!("func data"),
            _ => (),
        };

        dest_addr
    }

    fn fill_type_of_for_struct_like_data(
        &mut self,
        id: Id,
        params: IdVec,
        dest_addr: CraneliftValue,
    ) {
        // []struct {
        //     name: string,
        //     type: Ty,
        // }

        let params = self
            .backend
            .semantic
            .parser
            .id_vec(params)
            .iter()
            .map(|&p| match self.backend.semantic.parser.nodes[p] {
                Node::DeclParam { name, ty, .. } => (self.backend.resolve_symbol(name), ty),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();

        // allocate array
        let array_ptr =
            self.declare_global_data_with_size(&format!("{}_typeof_struct_array", id), 16);

        // set array len
        let len = self.builder.ins().iconst(types::I64, params.len() as i64);
        self.builder.ins().store(MemFlags::new(), len, array_ptr, 0);

        // allocate array data ptr
        let size_of_struct = 24; // 16 for `name: string`, 8 for `type: *Ty`
        let array_data_ptr = self.declare_global_data_with_size(
            &format!("{}_typeof_struct_array_data", id),
            size_of_struct * params.len() as u32,
        );

        // set array data ptr
        self.builder
            .ins()
            .store(MemFlags::new(), array_data_ptr, array_ptr, 8);

        let mut offset = 0;
        for (name, ty) in params {
            // store name length
            let name_len = self
                .builder
                .ins()
                .iconst(types::I64, name.bytes().len() as i64);
            self.builder
                .ins()
                .store(MemFlags::new(), name_len, array_data_ptr, offset);
            offset += 8;

            // store the name ptr
            let name_ptr = self.declare_global_data_with_bytes(
                &format!("param_name_{}_{}", id, name),
                name.into_boxed_str().into_boxed_bytes(),
            );
            self.builder
                .ins()
                .store(MemFlags::new(), name_ptr, array_data_ptr, offset);
            offset += 8;

            // store ty
            let ty_ptr = self.get_global_ptr_to_type_of(ty);
            self.builder
                .ins()
                .store(MemFlags::new(), ty_ptr, array_data_ptr, offset);
            offset += 8;
        }

        // store array into dest_addr
        let dest_addr = self
            .builder
            .ins()
            .iadd_imm(dest_addr, ENUM_TAG_SIZE_BYTES as i64);
        self.builder.emit_small_memcpy(
            self.backend.module.isa().frontend_config(),
            dest_addr,
            array_ptr,
            16,
            1,
            1,
        );
    }

    fn compile_call(&mut self, id: Id, name: Id, params: &[Id]) -> Result<(), CompileError> {
        // todo(chad): This doesn't need to be dynamic (call and then call-indirect) when compiling straight to an object file

        self.compile_id(name)?;
        for param in params {
            self.compile_id(*param)?;
        }

        let func_ty = self.backend.semantic.types[name];
        let return_ty = match func_ty {
            Type::Func { return_ty, .. } => Some(return_ty),
            _ => None,
        }
        .unwrap();
        let return_size = type_size(&self.backend.semantic, &self.backend.module, return_ty);

        // direct call?
        if let Value::FuncRef(fr) = self.backend.values[name] {
            let cranelift_params = params
                .iter()
                .map(|param| self.rvalue(*param))
                .collect::<Vec<_>>();

            let call_inst = self.builder.ins().call(fr, &cranelift_params);
            if return_size > 0 {
                let value = self.builder.func.dfg.inst_results(call_inst)[0];
                self.set_value(id, value);
            }

            return Ok(());
        }

        let cranelift_param = match &self.backend.values[name] {
            Value::FuncSym(fr) => self.builder.ins().iconst(types::I64, fr.to_usize() as i64),
            Value::Value(_) => self.rvalue(name),
            _ => unreachable!("unrecognized Value type in codegen Node::Call"),
        };

        let call_inst = self
            .builder
            .ins()
            .call(self.dynamic_fn_ptr_decl, &[cranelift_param]);
        let value = self.builder.func.dfg.inst_results(call_inst)[0];

        // Now call indirect
        let mut sig = self.backend.module.make_signature();
        for param in params.iter() {
            sig.params
                .push(AbiParam::new(into_cranelift_type(&self.backend, *param)?));
        }

        if return_size > 0 {
            sig.returns.push(AbiParam::new(into_cranelift_type(
                &self.backend,
                return_ty,
            )?));
        }

        let cranelift_params = params
            .iter()
            .map(|param| self.rvalue(*param))
            .collect::<Vec<_>>();

        let sig = self.builder.import_signature(sig);
        let call_inst = self
            .builder
            .ins()
            .call_indirect(sig, value, &cranelift_params);

        if return_size > 0 {
            let value = self.builder.func.dfg.inst_results(call_inst)[0];
            self.set_value(id, value);
        }

        Ok(())
    }

    fn load(&mut self, ty: types::Type, value: CraneliftValue, offset: i32) -> CraneliftValue {
        // dbg!(ty);
        self.builder.ins().load(ty, MemFlags::new(), value, offset)
    }
}

pub fn into_cranelift_type(backend: &Backend, id: Id) -> Result<types::Type, CompileError> {
    // if backend.rvalue_is_ptr(id) {
    //     return Ok(types::I64); // todo(chad): isa ptr type
    // }

    let range = backend.semantic.parser.ranges[id];
    let ty = &backend.semantic.types[id];

    match ty {
        Type::Basic(bt) => match bt {
            BasicType::Bool => Ok(types::I32), // todo(chad): I8?
            BasicType::I8 => Ok(types::I8),
            BasicType::I16 => Ok(types::I16),
            BasicType::I32 => Ok(types::I32),
            BasicType::I64 => Ok(types::I64),
            BasicType::F32 => Ok(types::F32),
            BasicType::F64 => Ok(types::F64),
            _ => Err(CompileError::from_string(
                format!("Could not convert type {:?}", ty),
                range,
            )),
        },
        Type::Func { .. } => Ok(types::I64),
        Type::Pointer(_) => Ok(types::I64), // todo(chad): need to get the actual type here, from the isa
        Type::Struct { params, .. } if backend.semantic.parser.id_vec(*params).is_empty() => {
            unreachable!()
        }
        _ => Err(CompileError::from_string(
            format!("Could not convert type {:?}", ty),
            range,
        )),
    }
}

pub fn type_size(semantic: &Semantic, module: &Module<SimpleJITBackend>, id: Id) -> u32 {
    match &semantic.types[id] {
        Type::Basic(BasicType::None) => 0,
        Type::Basic(BasicType::Bool) => 4,
        Type::Basic(BasicType::I8) => 1,
        Type::Basic(BasicType::I16) => 2,
        Type::Basic(BasicType::I32) => 4,
        Type::Basic(BasicType::I64) => 8,
        Type::Basic(BasicType::F32) => 4,
        Type::Basic(BasicType::F64) => 8,
        Type::Pointer(_) => module.isa().pointer_bytes() as _,
        Type::Func { .. } => module.isa().pointer_bytes() as _,
        Type::Struct { params, .. } | Type::StructLiteral(params) => semantic
            .parser
            .id_vec(*params)
            .iter()
            .map(|p| type_size(semantic, module, *p))
            .sum(),
        Type::Enum { params, .. } => {
            let biggest_param = semantic
                .parser
                .id_vec(*params)
                .iter()
                .map(|p| type_size(semantic, module, *p))
                .max()
                .unwrap_or(0);
            let tag_size = ENUM_TAG_SIZE_BYTES;
            tag_size + biggest_param
        }
        // 64-bit length + ptr
        Type::Array(_) | Type::String => module.isa().pointer_bytes() as u32 + 8,
        _ => todo!(
            "type_size for {:?} ({:?}) with node id {} at {:?}",
            &semantic.types[id],
            // semantic.parser.debug(id),
            semantic.parser.nodes[id],
            id,
            semantic.parser.ranges[id]
        ),
    }
}

fn declare_externs<'a, 'b, 'c>(
    backend: &mut Backend<'a, 'b>,
    builder: &mut FunctionBuilder<'c>,
) -> Result<(), CompileError> {
    for ext in backend.semantic.parser.externs.clone() {
        let mut sig = backend.module.make_signature();

        let (name, params, return_ty) = match backend.semantic.parser.nodes[ext] {
            Node::Extern {
                name,
                params,
                return_ty,
            } => (name, params, return_ty),
            _ => unreachable!(),
        };

        let return_size = type_size(backend.semantic, &backend.module, return_ty);
        if return_size > 0 {
            sig.returns
                .push(AbiParam::new(into_cranelift_type(backend, return_ty)?));
        }

        for param in backend.semantic.parser.id_vec(params).clone() {
            sig.params
                .push(AbiParam::new(into_cranelift_type(backend, param)?));
        }

        let name = backend
            .semantic
            .parser
            .lexer
            .string_interner
            .resolve(name)
            .unwrap()
            .to_string();

        let func_id = backend
            .module
            .declare_function(&name, Linkage::Import, &sig)
            .unwrap();

        backend.values[ext] = Value::FuncRef(
            backend
                .module
                .declare_func_in_func(func_id, &mut builder.func),
        );
    }

    Ok(())
}

fn dynamic_fn_ptr(sym: Sym) -> *const u8 {
    FUNC_PTRS.lock().unwrap().get(&sym).unwrap().0
}

fn print_i8(n: i8) {
    print!("{}", n);
}

fn print_i16(n: i16) {
    print!("{}", n);
}

fn print_i32(n: i32) {
    print!("{}", n);
}

fn print_i64(n: i64) {
    print!("{}", n);
}

fn print_string(len: i64, bytes: *mut u8) {
    let s = unsafe { String::from_raw_parts(bytes, len as usize, len as usize) };
    print!("{}", s);
    std::mem::forget(s);
}

fn alloc_helper(bytes: i64) -> *mut u8 {
    unsafe { std::alloc::alloc_zeroed(std::alloc::Layout::from_size_align(bytes as _, 1).unwrap()) }
}

#[repr(C)]
struct BackendToken {
    tag: i16,
    data: [u8; 16],
}

#[repr(C)]
struct BackendTokenArray {
    length: i64,
    data: *mut BackendToken,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BackendString {
    len: i64,
    data: *mut u8,
}

fn push_token(semantic: *mut Semantic, tokens: i64, token: *const BackendToken) {
    let semantic: &mut Semantic = unsafe { std::mem::transmute(semantic) };

    let tag = unsafe { (*token).tag };

    let tokens = semantic.parser.token_vecs.get_mut(tokens as usize).unwrap();

    match tag {
        0 => {
            tokens.push(Lexeme::new(Token::LParen, Range::default()));
        } // LParen,
        1 => {
            tokens.push(Lexeme::new(Token::RParen, Range::default()));
        } // RParen,
        2 => {
            tokens.push(Lexeme::new(Token::LCurly, Range::default()));
        } // LCurly,
        3 => {
            tokens.push(Lexeme::new(Token::RCurly, Range::default()));
        } // RCurly,
        4 => {
            tokens.push(Lexeme::new(Token::LSquare, Range::default()));
        } // LSquare,
        5 => {
            tokens.push(Lexeme::new(Token::RSquare, Range::default()));
        } // RSquare,
        6 => {
            tokens.push(Lexeme::new(Token::DoubleLAngle, Range::default()));
        } // DoubleLAngle,
        7 => {
            tokens.push(Lexeme::new(Token::DoubleRAngle, Range::default()));
        } // DoubleRAngle,
        8 => {
            tokens.push(Lexeme::new(Token::LAngle, Range::default()));
        } // LAngle,
        9 => {
            tokens.push(Lexeme::new(Token::RAngle, Range::default()));
        } // RAngle,
        10 => {
            tokens.push(Lexeme::new(Token::Return, Range::default()));
        } // RAngle,
        11 => {
            tokens.push(Lexeme::new(Token::Semicolon, Range::default()));
        } // Semicolon,
        12 => {
            tokens.push(Lexeme::new(Token::Colon, Range::default()));
        } // Colon,
        13 => {
            tokens.push(Lexeme::new(Token::Bang, Range::default()));
        } // Bang,
        14 => {
            tokens.push(Lexeme::new(Token::Dot, Range::default()));
        } // Dot,
        15 => {
            tokens.push(Lexeme::new(Token::Asterisk, Range::default()));
        } // Asterisk,
        16 => {
            tokens.push(Lexeme::new(Token::EqEq, Range::default()));
        } // EqEq,
        17 => {
            tokens.push(Lexeme::new(Token::Neq, Range::default()));
        } // Neq,
        18 => {
            tokens.push(Lexeme::new(Token::Add, Range::default()));
        } // Add,
        19 => {
            tokens.push(Lexeme::new(Token::Sub, Range::default()));
        } // Sub,
        20 => {
            tokens.push(Lexeme::new(Token::Mul, Range::default()));
        } // Mul,
        21 => {
            tokens.push(Lexeme::new(Token::Div, Range::default()));
        } // Div,
        22 => {
            tokens.push(Lexeme::new(Token::Eq, Range::default()));
        } // Eq,
        23 => {
            tokens.push(Lexeme::new(Token::Comma, Range::default()));
        } // Comma,
        24 => {
            tokens.push(Lexeme::new(Token::Struct, Range::default()));
        } // Struct,
        25 => {
            tokens.push(Lexeme::new(Token::Enum, Range::default()));
        } // Enum,
        26 => {
            tokens.push(Lexeme::new(Token::Fn, Range::default()));
        } // Fn,
        27 => {
            tokens.push(Lexeme::new(Token::Macro, Range::default()));
        } // Macro,
        28 => {
            tokens.push(Lexeme::new(Token::Extern, Range::default()));
        } // Extern,
        29 => {
            tokens.push(Lexeme::new(Token::TypeOf, Range::default()));
        } // TypeOf,
        30 => {
            tokens.push(Lexeme::new(Token::Cast, Range::default()));
        } // Cast,
        31 => {
            tokens.push(Lexeme::new(Token::If, Range::default()));
        } // If,
        32 => {
            tokens.push(Lexeme::new(Token::While, Range::default()));
        } // While,
        33 => {
            tokens.push(Lexeme::new(Token::Else, Range::default()));
        } // Else,
        34 => {
            tokens.push(Lexeme::new(Token::True, Range::default()));
        } // True,
        35 => {
            tokens.push(Lexeme::new(Token::False, Range::default()));
        } // False,
        36 => {
            tokens.push(Lexeme::new(Token::I8, Range::default()));
        } // I8,
        37 => {
            tokens.push(Lexeme::new(Token::I16, Range::default()));
        } // I16,
        38 => {
            tokens.push(Lexeme::new(Token::I32, Range::default()));
        } // I32,
        39 => {
            tokens.push(Lexeme::new(Token::I64, Range::default()));
        } // I64,
        40 => {
            tokens.push(Lexeme::new(Token::Ampersand, Range::default()));
        } // Ampersand,
        41 => {
            tokens.push(Lexeme::new(Token::Type, Range::default()));
        } // Type_,
        42 => {
            tokens.push(Lexeme::new(Token::Caret, Range::default()));
        } // Caret,
        43 => {
            tokens.push(Lexeme::new(Token::Uninit, Range::default()));
        } // Uninit,
        44 => {
            let sym = unsafe {
                let bs = *std::mem::transmute::<&[u8; 16], *const BackendString>(&((*token).data));
                let s =
                    String::from_raw_parts(bs.data as *mut u8, bs.len as usize, bs.len as usize);

                let sym = semantic
                    .parser
                    .lexer
                    .string_interner
                    .get_or_intern(s.clone());
                std::mem::forget(s);
                sym
            };

            tokens.push(Lexeme::new(Token::Symbol(sym), Range::default()));
        } // Symbol: string,
        45 => {
            let n = unsafe { *std::mem::transmute::<&[u8; 16], *const i64>(&((*token).data)) };
            tokens.push(Lexeme::new(
                Token::IntegerLiteral(n, NumericSpecification::I64),
                Range::default(),
            ));
        } // IntegerLiteral: i64, // todo(chad): specification
        46 => {
            let n = unsafe { *std::mem::transmute::<&[u8; 16], *const f64>(&((*token).data)) };
            tokens.push(Lexeme::new(Token::FloatLiteral(n), Range::default()));
        } // FloatLiteral: f64, // todo(chad): specification
        47 => {
            let sym = unsafe {
                let bs = *std::mem::transmute::<&[u8; 16], *const BackendString>(&((*token).data));
                let s =
                    String::from_raw_parts(bs.data as *mut u8, bs.len as usize, bs.len as usize);

                let sym = semantic
                    .parser
                    .lexer
                    .string_interner
                    .get_or_intern(s.clone());
                std::mem::forget(s);
                sym
            };

            tokens.push(Lexeme::new(Token::StringLiteral(sym), Range::default()));
        } // StringLiteral: string,
        _ => unreachable!(),
    }
}
