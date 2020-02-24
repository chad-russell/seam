use crate::parser::{BasicType, CompileError, Id, Lexeme, Node, Parser, Range, Token, Type};
use crate::semantic::Semantic;

use cranelift::codegen::ir::{FuncRef, MemFlags, Signature, StackSlotData, StackSlotKind};
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
    pub values: Vec<Value>,
    pub funcs: BTreeMap<Sym, FuncId>,
    pub func_sigs: BTreeMap<Sym, Signature>,
}

impl<'a, 'b> Backend<'a, 'b> {
    pub fn new(semantic: &'a mut Semantic<'b>) -> Self {
        let len = semantic.types.len();

        Self {
            semantic,
            values: vec![Value::Unassigned; len],
            funcs: BTreeMap::new(),
            func_sigs: BTreeMap::new(),
        }
    }

    #[allow(dead_code)]
    pub fn update_source(&mut self, source: &'b str) -> Result<(), CompileError> {
        self.semantic.parser.top_level.clear();
        self.semantic.parser.top_level_map.clear();
        self.semantic.parser.nodes.clear();
        self.semantic.parser.node_scopes.clear();
        self.semantic.parser.ranges.clear();
        self.semantic.parser.scopes.clear();

        self.semantic.parser.lexer.update_source(source);

        self.semantic.parser.parse()?;

        self.semantic.topo.clear();
        self.semantic.types.clear();
        while self.semantic.types.len() < self.semantic.parser.nodes.len() {
            self.semantic.types.push(Type::Unassigned);
        }

        self.assign_top_level_types()?;

        Ok(())
    }

    pub fn generate_macro_call_order(&self, call_id: Id, mac_topo: &mut Vec<Id>) {
        let macro_name = match self.semantic.parser.nodes[call_id] {
            Node::MacroCall { name, .. } => name,
            _ => unreachable!(),
        };
        let resolved_macro = self.semantic.resolve(macro_name).unwrap();
        let resolved_scope = self.semantic.parser.node_scopes[resolved_macro];

        // let func = self.semantic.parser.function_by_macro_call[&resolved_scope];
        let calls_before = self
            .semantic
            .parser
            .macro_calls_by_function
            .get(&resolved_scope);

        if let Some(calls_before) = calls_before {
            for &cb in calls_before {
                if cb != call_id {
                    self.generate_macro_call_order(cb, mac_topo);
                }
            }
        }

        if !mac_topo.contains(&call_id) {
            mac_topo.push(call_id);
        }
    }

    pub fn assign_top_level_types(&mut self) -> Result<(), CompileError> {
        // ugh clone...
        for t in self.semantic.parser.externs.clone() {
            self.semantic.assign_type(t)?;
            self.compile_id(t)?;
        }

        let mut mac_topo = Vec::new();
        for t in self.semantic.parser.macro_calls.clone() {
            self.generate_macro_call_order(t, &mut mac_topo);
        }

        for t in mac_topo {
            self.semantic.assign_type(t)?;
            for t in self.semantic.topo.clone() {
                self.semantic.assign_type(t)?;
            }
            self.semantic.unify_types()?;
            for t in self.semantic.topo.clone() {
                self.compile_id(t)?;
            }
            self.semantic.topo.clear();
            self.compile_id(t)?;
        }

        // ugh clone...
        for tl in self.semantic.parser.top_level.clone() {
            let is_poly = match &self.semantic.parser.nodes[tl] {
                Node::Func { ct_params, .. } => ct_params.is_some(),
                Node::Struct { ct_params, .. } => ct_params.is_some(),
                Node::Enum { ct_params, .. } => ct_params.is_some(),
                _ => false,
            };

            if !is_poly {
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

    // pub fn recompile_function(&mut self, function: impl Into<String>) -> Result<(), CompileError> {
    //     self.values.clear();
    //     while self.values.len() < self.semantic.parser.nodes.len() {
    //         self.values.push(Value::Unassigned);
    //     }

    //     // Replace all top-level function values with FuncRefs
    //     let new_id = self.semantic.parser.get_top_level(function).unwrap();
    //     for topo in self.semantic.topo.clone() {
    //         if topo == new_id {
    //             continue;
    //         }

    //         match self.semantic.parser.nodes[topo] {
    //             Node::Func { name, .. } => {
    //                 self.values[topo] = Value::FuncSym(name);
    //             }
    //             _ => {}
    //         };
    //     }

    //     self.compile_id(new_id)?;

    //     Ok(())
    // }

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
            std::mem::transmute::<_, fn() -> i64>(
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
                is_macro,
                ..
            } => {
                // todo(chad): find a way to reuse these
                let mut module: Module<SimpleJITBackend> = {
                    let mut jit_builder = SimpleJITBuilder::new(default_libcall_names());
                    jit_builder.symbol("__dynamic_fn_ptr", dynamic_fn_ptr as *const u8);
                    // jit_builder.symbol("__macro_insert", macro_insert as *const u8);
                    // jit_builder.symbol("__prepare_unquote", prepare_unquote as *const u8);
                    jit_builder.symbol("print_int", print_int as *const u8);
                    jit_builder.symbol("print_string", print_string as *const u8);
                    for (&name, &ptr) in FUNC_PTRS.lock().unwrap().iter() {
                        jit_builder.symbol(self.resolve_symbol(name), ptr.0);
                    }

                    Module::new(jit_builder)
                };

                let string_literal_data_ids = self.get_string_literal_data_ids(&mut module);

                self.funcs.clear();
                for (name, sig) in self.func_sigs.iter() {
                    self.funcs.insert(
                        name.clone(),
                        module
                            .declare_function(&self.resolve_symbol(*name), Linkage::Import, &sig)
                            .unwrap(),
                    );
                }

                let mut ctx = module.make_context();

                let func_name = String::from(self.semantic.parser.resolve_sym_unchecked(name));
                let func_sym = self.get_symbol(func_name.clone());

                self.values[id] = Value::FuncSym(func_sym);

                // println!("compiling func {}", func_name);

                let mut sig = module.make_signature();

                for param in self.semantic.parser.id_vec(params) {
                    sig.params
                        .push(AbiParam::new(into_cranelift_type(&self.semantic, *param)?));
                }
                if is_macro {
                    // macros always take a pointer to a `Semantic` as their last parameter
                    sig.params.push(AbiParam::new(types::I64));
                }

                let return_size = type_size(&self.semantic, &module, return_ty);
                if return_size > 0 {
                    sig.returns.push(AbiParam::new(into_cranelift_type(
                        &self.semantic,
                        return_ty,
                    )?));
                }

                ctx.func.signature = sig;

                self.func_sigs.insert(func_sym, ctx.func.signature.clone());

                let func = module
                    .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
                    .unwrap();
                ctx.func.name = ExternalName::user(0, func.as_u32());

                let mut func_ctx = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
                let ebb = builder.create_ebb();
                builder.switch_to_block(ebb);
                builder.append_ebb_params_for_function_params(ebb);

                let dfp_decl = {
                    let mut dfp_sig = module.make_signature();

                    dfp_sig.params.push(AbiParam::new(types::I64));
                    dfp_sig
                        .returns
                        .push(AbiParam::new(module.isa().pointer_type()));

                    let dfp_func_id = module
                        .declare_function("__dynamic_fn_ptr", Linkage::Import, &dfp_sig)
                        .unwrap();

                    module.declare_func_in_func(dfp_func_id, &mut builder.func)
                };

                let insert_decl = {
                    let mut ins_sig = module.make_signature();

                    ins_sig
                        .params
                        .push(AbiParam::new(module.isa().pointer_type()));
                    ins_sig.params.push(AbiParam::new(types::I64));
                    ins_sig
                        .returns
                        .push(AbiParam::new(module.isa().pointer_type()));

                    let ins_func_id = module
                        .declare_function("__macro_insert", Linkage::Import, &ins_sig)
                        .unwrap();

                    module.declare_func_in_func(ins_func_id, &mut builder.func)
                };

                // todo(chad): make all these decls part of a function instead of declaring them twice
                let prepare_unquote_decl = {
                    let mut sig = module.make_signature();

                    sig.params.push(AbiParam::new(module.isa().pointer_type()));
                    sig.params.push(AbiParam::new(types::I64)); // 1 if unquoting a 'Code', 0 otherwise
                    sig.params.push(AbiParam::new(types::I64));
                    sig.params.push(AbiParam::new(module.isa().pointer_type()));

                    let func_id = module
                        .declare_function("__prepare_unquote", Linkage::Import, &sig)
                        .unwrap();

                    module.declare_func_in_func(func_id, &mut builder.func)
                };

                declare_externs(&self.semantic, &mut module, &mut builder, &mut self.values)?;

                let mut fb = FunctionBackend {
                    backend: self,
                    builder,
                    current_block: ebb,
                    string_literal_data_ids,
                    dynamic_fn_ptr_decl: dfp_decl,
                    prepare_unquote_decl,
                    insert_decl,
                    module: &mut module,
                };

                for stmt in fb.backend.semantic.parser.id_vec(stmts).clone() {
                    fb.compile_id(stmt)?;
                }

                fb.builder.seal_all_blocks();
                fb.builder.finalize();

                // println!("{}", ctx.func.display(None));

                module.define_function(func, &mut ctx).unwrap();
                module.clear_context(&mut ctx);

                module.finalize_definitions();

                let func = module.get_finalized_function(func);
                FUNC_PTRS.lock().unwrap().insert(func_sym, FuncPtr(func));

                Ok(())
            }
            Node::Symbol(sym) => {
                let resolved = self.semantic.scope_get(sym, id)?;
                self.compile_id(resolved)?;
                self.semantic.parser.node_is_addressable[id] =
                    self.semantic.parser.node_is_addressable[resolved];
                self.values[id] = self.values[resolved].clone();

                Ok(())
            }
            Node::MacroCall {
                name,
                params,
                expanded: _,
            } => {
                self.values[id] = Value::None;

                self.compile_id(name)?;

                let params = self.semantic.parser.id_vec(params).clone();
                let name = match self.semantic.parser.nodes[name] {
                    Node::Symbol(sym) => sym,
                    _ => unreachable!(),
                };

                // todo(chad): if the macro is Tokens -> Tokens with no other params, we should NOT build this extra layer
                let compiled_macro = self.build_hoisted_function(params, name)?;
                let compiled_macro: fn(semantic: *mut Semantic) -> i64 =
                    unsafe { std::mem::transmute(compiled_macro) };
                let tokens_id = compiled_macro(self.semantic as *mut _);
                let tokens = self.semantic.parser.token_vecs[tokens_id as usize].clone();

                self.semantic.parser.lexer.macro_tokens = Some(tokens);
                self.semantic.parser.top_scope = self.semantic.parser.node_scopes[id];
                self.semantic.parser.lexer.top = Lexeme::new(Token::EOF, Range::default());
                self.semantic.parser.lexer.second = Lexeme::new(Token::EOF, Range::default());
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

    pub fn allocate_for_new_nodes(&mut self) {
        while self.values.len() < self.semantic.parser.nodes.len() {
            self.values.push(Value::Unassigned);
        }
    }

    pub fn build_hoisted_function(
        &mut self,
        params: Vec<Id>,
        hoisting_name: Sym,
    ) -> Result<*const u8, CompileError> {
        // println!(
        //     "compiling hoisting function for {}",
        //     self.resolve_symbol(hoisting_name)
        // );

        let mut module: Module<SimpleJITBackend> = {
            let mut jit_builder = SimpleJITBuilder::new(default_libcall_names());
            jit_builder.symbol("__dynamic_fn_ptr", dynamic_fn_ptr as *const u8);
            jit_builder.symbol("print_int", print_int as *const u8);
            jit_builder.symbol("print_string", print_string as *const u8);
            for (&name, &ptr) in crate::backend::FUNC_PTRS.lock().unwrap().iter() {
                jit_builder.symbol(self.resolve_symbol(name), ptr.0);
            }

            Module::new(jit_builder)
        };

        let string_literal_data_ids = self.get_string_literal_data_ids(&mut module);

        self.funcs.clear();
        for (name, sig) in self.func_sigs.iter() {
            self.funcs.insert(
                name.clone(),
                module
                    .declare_function(&self.resolve_symbol(*name), Linkage::Import, &sig)
                    .unwrap(),
            );
        }

        let mut ctx = module.make_context();
        let mut sig = module.make_signature();

        // param 0 = ptr to semantic
        // todo(chad): eventually param 1 = ptr to return value
        sig.params.push(AbiParam::new(module.isa().pointer_type()));
        sig.returns.push(AbiParam::new(types::I64));

        ctx.func.signature = sig;

        // self
        //     .func_sigs
        //     .insert(func_sym, ctx.func.signature.clone());

        let func = module
            .declare_function("hoist", Linkage::Local, &ctx.func.signature)
            .unwrap();
        ctx.func.name = ExternalName::user(0, func.as_u32());

        let dfp_decl = {
            let mut dfp_sig = module.make_signature();

            dfp_sig.params.push(AbiParam::new(types::I64));
            dfp_sig
                .returns
                .push(AbiParam::new(module.isa().pointer_type()));

            let dfp_func_id = module
                .declare_function("__dynamic_fn_ptr", Linkage::Import, &dfp_sig)
                .unwrap();

            module.declare_func_in_func(dfp_func_id, &mut ctx.func)
        };

        let insert_decl = {
            let mut ins_sig = module.make_signature();

            ins_sig.params.push(AbiParam::new(types::I64));
            ins_sig.params.push(AbiParam::new(types::I64));
            ins_sig
                .returns
                .push(AbiParam::new(module.isa().pointer_type()));

            let ins_func_id = module
                .declare_function("__macro_insert", Linkage::Import, &ins_sig)
                .unwrap();

            module.declare_func_in_func(ins_func_id, &mut ctx.func)
        };

        // todo(chad): make all these decls part of a function instead of declaring them twice
        let prepare_unquote_decl = {
            let mut sig = module.make_signature();

            sig.params.push(AbiParam::new(module.isa().pointer_type()));
            sig.params.push(AbiParam::new(types::I64));
            sig.params.push(AbiParam::new(module.isa().pointer_type()));

            let func_id = module
                .declare_function("__prepare_unquote", Linkage::Import, &sig)
                .unwrap();

            module.declare_func_in_func(func_id, &mut ctx.func)
        };

        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

        declare_externs(&self.semantic, &mut module, &mut builder, &mut self.values)?;

        let ebb = builder.create_ebb();
        builder.switch_to_block(ebb);
        builder.append_ebb_params_for_function_params(ebb);

        let mut fb = FunctionBackend {
            backend: self,
            builder,
            current_block: ebb,
            string_literal_data_ids,
            dynamic_fn_ptr_decl: dfp_decl,
            insert_decl,
            prepare_unquote_decl,
            module: &mut module,
        };

        for &param in params.iter() {
            fb.compile_id(param)?;
        }

        // make the call
        let magic_param = {
            let current_fn_params = fb.builder.ebb_params(fb.current_block);
            *current_fn_params.last().unwrap()
        };
        let normal_params = params
            .iter()
            .map(|param| fb.rvalue(*param))
            .collect::<Vec<_>>();
        let mut cranelift_params = normal_params;
        cranelift_params.push(magic_param);

        let found_macro = fb.backend.funcs[&hoisting_name];
        let found_macro = fb
            .module
            .declare_func_in_func(found_macro, &mut fb.builder.func);

        let call_inst = fb.builder.ins().call(found_macro, &cranelift_params);
        let return_value = fb.builder.func.dfg.inst_results(call_inst)[0];

        fb.builder.ins().return_(&[return_value]);

        fb.builder.seal_all_blocks();
        fb.builder.finalize();

        // println!("{}", ctx.func.display(None));

        module.define_function(func, &mut ctx).unwrap();
        module.clear_context(&mut ctx);

        module.finalize_definitions();

        Ok(module.get_finalized_function(func))
    }

    fn get_string_literal_data_ids(
        &mut self,
        module: &mut Module<SimpleJITBackend>,
    ) -> BTreeMap<Id, DataId> {
        let mut string_literal_data_ids = BTreeMap::new();

        for lit in self.semantic.parser.string_literals.clone() {
            // declare data section for string literals
            let string_literal_data_id = module
                .declare_data(&format!("str_lit_{}", lit), Linkage::Local, false, None)
                .unwrap();

            let mut data_ctx = DataContext::new();
            let mut lits = String::new();
            let lit_sym = match self.semantic.parser.nodes[lit] {
                Node::StringLiteral { sym, .. } => sym,
                _ => unreachable!(),
            };
            let lit_str = self.resolve_symbol(lit_sym);

            lits += lit_str.as_str();

            let raw_bytes: Box<[u8]> = lits.into_boxed_str().into_boxed_bytes();
            data_ctx.define(raw_bytes);

            module
                .define_data(string_literal_data_id, &data_ctx)
                .unwrap();

            string_literal_data_ids.insert(lit, string_literal_data_id);
        }

        string_literal_data_ids
    }
}

// fn get_cranelift_type(semantic: &Semantic, id: Id) -> Result<types::Type, CompileError> {
//     let range = semantic.parser.ranges[id];

//     let ty = semantic
//         .types
//         .get(id)
//         .ok_or(CompileError::from_string("Type not found", range))?;

//         .try_into()
//         .map_err(|e| CompileError::from_string(String::from(e), range))
// }

pub struct FunctionBackend<'a, 'b, 'c, 'd> {
    pub backend: &'a mut Backend<'b, 'c>,
    pub builder: FunctionBuilder<'d>,
    pub current_block: Ebb,
    pub string_literal_data_ids: BTreeMap<Id, DataId>,
    pub dynamic_fn_ptr_decl: FuncRef,
    pub prepare_unquote_decl: FuncRef,
    pub insert_decl: FuncRef,
    pub module: &'a mut Module<SimpleJITBackend>,
}

impl<'a, 'b, 'c, 'd> FunctionBackend<'a, 'b, 'c, 'd> {
    fn set_value(&mut self, id: Id, value: CraneliftValue) {
        self.backend.values[id] = Value::Value(value);
    }

    // Returns whether the value for a particular id is the value itself, or a pointer to the value
    // For instance, the constant integer 3 would likely be just the value, and not a pointer.
    // However a field access (or even a load) needs to always be returned by pointer, because it might refer to a struct literal.
    // Some nodes (like const int above) don't usually return a pointer, but they might need to for other reasons,
    // e.g. they need their address to be taken. In that case we store that fact in the 'node_is_addressable' vec.
    // Also symbols won't know whether they have a local or not because it depends what they're referencing. So 'node_is_addressable' is handy there too.
    fn rvalue_is_ptr(&mut self, id: Id) -> bool {
        if self.backend.semantic.parser.node_is_addressable[id] {
            return true;
        }

        match self.backend.semantic.parser.nodes[id] {
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

    fn as_value(&mut self, id: Id) -> CraneliftValue {
        match self.backend.values[id] {
            Value::Value(v) => Some(v),
            Value::FuncSym(sym) => Some(self.builder.ins().iconst(types::I64, sym as i64)),
            _ => None,
        }
        .expect(&format!(
            "Cannot convert {:?} into a CraneliftValue",
            &self.backend.values[id]
        ))
    }

    fn store(&mut self, id: Id, dest: &Value) {
        if self.rvalue_is_ptr(id) {
            self.store_copy(id, dest);
        } else {
            self.store_value(id, dest, None);
        }
    }

    fn store_with_offset(&mut self, id: Id, dest: &Value, offset: i32) {
        if self.rvalue_is_ptr(id) {
            let dest = if offset == 0 {
                *dest
            } else {
                let offset = self.builder.ins().iconst(types::I64, offset as i64);
                let dest = dest.as_value_relaxed(self);
                let dest = self.builder.ins().iadd(dest, offset);
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
        let size = type_size(&self.backend.semantic, &self.module, id);

        let source_value = self.as_value(id);
        let dest_value = dest.as_value_relaxed(self);

        self.builder.emit_small_memcpy(
            self.module.isa().frontend_config(),
            dest_value,
            source_value,
            size as _,
            1,
            1,
        );
    }

    fn rvalue(&mut self, id: Id) -> CraneliftValue {
        if self.rvalue_is_ptr(id) {
            let ty = &self.backend.semantic.types[id];
            let ty = match ty {
                Type::Struct { .. } => self.module.isa().pointer_type(),
                Type::Enum { .. } => self.module.isa().pointer_type(),
                _ => into_cranelift_type(&self.backend.semantic, id).unwrap(),
            };

            match &self.backend.values[id] {
                Value::Value(v) => self.builder.ins().load(ty, MemFlags::new(), *v, 0),
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

                let return_size = type_size(&self.backend.semantic, &self.module, return_id);
                if return_size > 0 {
                    self.builder.ins().return_(&[value]);
                } else {
                    self.builder.ins().return_(&[]);
                }

                Ok(())
            }
            Node::IntLiteral(i) => {
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
                    let size = type_size(&self.backend.semantic, &self.module, id);
                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                        offset: None,
                    });
                    let slot_addr =
                        self.builder
                            .ins()
                            .stack_addr(self.module.isa().pointer_type(), slot, 0);
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
                match &self.backend.semantic.types[id] {
                    Type::Enum { .. } => {
                        let params = self.backend.semantic.parser.id_vec(params);

                        assert_eq!(params.len(), 1);
                        let param = params[0];
                        let size = type_size(&self.backend.semantic, &self.module, param)
                            + ENUM_TAG_SIZE_BYTES;

                        let slot = self.builder.create_stack_slot(StackSlotData {
                            kind: StackSlotKind::ExplicitSlot,
                            size,
                            offset: None,
                        });
                        let addr = self.builder.ins().stack_addr(
                            self.module.isa().pointer_type(),
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
                    _ => (),
                };

                // todo(chad): @Optimization: this is about the slowest thing we could ever do, but works great.
                // Come back later once everything is working and make it fast
                let size = type_size(&self.backend.semantic, &self.module, id);
                let slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let addr = self
                    .builder
                    .ins()
                    .stack_addr(self.module.isa().pointer_type(), slot, 0);

                self.backend.values[id] = Value::Value(addr);

                let mut offset: i32 = 0;
                for param in self.backend.semantic.parser.id_vec(params).clone() {
                    self.compile_id(param)?;
                    self.store_with_offset(param, &Value::Value(addr), offset);
                    offset += type_size(&self.backend.semantic, &self.module, param) as i32;
                }

                Ok(())
            }
            Node::Symbol(sym) => {
                let resolved = self.backend.semantic.scope_get(sym, id)?;
                self.compile_id(resolved)?;
                self.backend.semantic.parser.node_is_addressable[id] =
                    self.backend.semantic.parser.node_is_addressable[resolved];
                self.backend.values[id] = self.backend.values[resolved].clone();

                Ok(())
            }
            // Node::Add(lhs, rhs) => {
            //     self.compile_id(*lhs)?;
            //     self.compile_id(*rhs)?;

            //     let lhs = self.rvalue(*lhs);
            //     let rhs = self.rvalue(*rhs);

            //     let value = self.builder.ins().iadd(lhs, rhs);
            //     self.set_value(id, value);

            //     Ok(())
            // }
            Node::Let {
                name: _, // Sym
                ty: _,   // Id
                expr,    // Id
            } => {
                let size = type_size(&self.backend.semantic, &self.module, id);
                let slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });

                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.module.isa().pointer_type(), slot, 0);
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
                    if self.rvalue_is_ptr(name) {
                        addr = self.builder.ins().load(
                            self.module.isa().pointer_type(),
                            MemFlags::new(),
                            addr,
                            0,
                        );
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

                let current_params = self.builder.ebb_params(self.current_block).to_vec().clone();

                self.builder.ins().brnz(cond, true_block, &current_params);
                self.builder.ins().jump(false_block, &current_params);

                // true
                {
                    self.builder.switch_to_block(true_block);
                    self.current_block = true_block;

                    for stmt in self.backend.semantic.parser.id_vec(true_stmts).clone() {
                        self.compile_id(stmt)?;
                    }

                    let current_params =
                        self.builder.ebb_params(self.current_block).to_vec().clone();
                    self.builder.ins().jump(cont_block, &current_params);
                }

                // false
                {
                    self.builder.switch_to_block(false_block);
                    self.current_block = false_block;

                    for stmt in self.backend.semantic.parser.id_vec(false_stmts).clone() {
                        self.compile_id(stmt)?;
                    }

                    let current_params =
                        self.builder.ebb_params(self.current_block).to_vec().clone();
                    self.builder.ins().jump(cont_block, &current_params);
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
                    let size = self.module.isa().pointer_bytes() as u32;
                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                        offset: None,
                    });
                    let slot_addr =
                        self.builder
                            .ins()
                            .stack_addr(self.module.isa().pointer_type(), slot, 0);
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
                    Type::Struct { .. } => self.module.isa().pointer_type(),
                    _ => into_cranelift_type(&self.backend.semantic, id)?,
                };

                let loaded = self.builder.ins().load(ty, MemFlags::new(), value, 0);

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
                    let size = type_size(&self.backend.semantic, &self.module, id);
                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size,
                        offset: None,
                    });

                    let slot_addr =
                        self.builder
                            .ins()
                            .stack_addr(self.module.isa().pointer_type(), slot, 0);
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
                match &self.backend.semantic.types[unpointered_ty] {
                    Type::String => {
                        let field_name = String::from(
                            self.backend
                                .semantic
                                .parser
                                .resolve_sym_unchecked(field_name),
                        );

                        let mut base = self.as_value(base);

                        if loaded {
                            // if we are doing field access through a pointer, do an extra load
                            base = self.builder.ins().load(
                                self.module.isa().pointer_type(),
                                MemFlags::new(),
                                base,
                                0,
                            );
                        }

                        match field_name.as_str() {
                            "len" => (),
                            "buf" => {
                                let offset = self.builder.ins().iconst(types::I64, 8);
                                base = self.builder.ins().iadd(base, offset);
                            }
                            _ => unreachable!(),
                        }

                        self.backend.values[id] = base.into();
                        return Ok(());
                    }
                    _ => (),
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
                        .map(|ty| type_size(&self.backend.semantic, &self.module, ty))
                        .sum::<u32>()
                };

                let mut base = self.as_value(base);

                if loaded {
                    // if we are doing field access through a pointer, do an extra load
                    base = self.builder.ins().load(
                        self.module.isa().pointer_type(),
                        MemFlags::new(),
                        base,
                        0,
                    );
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
            Node::Tokens(token_vec_id) => {
                let value = self.builder.ins().iconst(types::I64, token_vec_id.0 as i64);

                self.set_value(id, value);

                if self.backend.semantic.parser.node_is_addressable[id] {
                    let slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: 8,
                        offset: None,
                    });
                    let slot_addr =
                        self.builder
                            .ins()
                            .stack_addr(self.module.isa().pointer_type(), slot, 0);
                    let value = Value::Value(slot_addr);
                    self.store_value(id, &value, None);
                    self.backend.values[id] = value;
                }

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
                    let element_size = type_size(&self.backend.semantic, &self.module, ty);
                    let struct_size =
                        type_size(&self.backend.semantic, &self.module, ty) * element_count as u32;
                    let struct_slot = self.builder.create_stack_slot(StackSlotData {
                        kind: StackSlotKind::ExplicitSlot,
                        size: struct_size,
                        offset: None,
                    });
                    let struct_slot_addr = self.builder.ins().stack_addr(
                        self.module.isa().pointer_type(),
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

                let size = type_size(&self.backend.semantic, &self.module, id);
                let struct_slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let dest_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.module.isa().pointer_type(), struct_slot, 0);

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
            Node::StringLiteral { bytes, .. } => {
                let lit_id = self.string_literal_data_ids[&id];

                let string_lit_ptr = self
                    .module
                    .declare_data_in_func(lit_id, &mut self.builder.func);
                let global = self
                    .builder
                    .ins()
                    .global_value(self.module.isa().pointer_type(), string_lit_ptr);

                let len_value = self.builder.ins().iconst(types::I64, bytes as i64);

                let size = type_size(&self.backend.semantic, &self.module, id);
                let struct_slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let dest_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.module.isa().pointer_type(), struct_slot, 0);

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
                let data_ptr = self.builder.ins().load(
                    self.module.isa().pointer_type(),
                    MemFlags::new(),
                    struct_ptr,
                    8,
                );

                let element_size = type_size(&self.backend.semantic, &self.module, id);
                let element_size = self.builder.ins().iconst(types::I64, element_size as i64);

                let index = self.rvalue(index);
                let offset = self.builder.ins().imul(element_size, index);

                self.backend.values[id] = Value::Value(self.builder.ins().iadd(data_ptr, offset));

                Ok(())
            }
            Node::TypeOf(expr) => {
                let expr_ty = self.backend.semantic.types[expr];

                // allocate struct
                let struct_size = type_size(
                    &self.backend.semantic,
                    &self.module,
                    self.backend.semantic.parser.ty_decl.unwrap(),
                );

                let tag = match expr_ty {
                    Type::Basic(BasicType::I8) => 0,
                    Type::Basic(BasicType::I16) => 1,
                    Type::Basic(BasicType::I64) => 2,
                    Type::Basic(BasicType::F32) => 3,
                    Type::Basic(BasicType::F64) => 4,
                    Type::String => 5,
                    Type::Pointer(_) => 6,
                    Type::Array(_) => 7,
                    Type::Struct { .. } => 8,
                    Type::Enum { .. } => 9,
                    Type::Func { .. } => 10,
                    _ => todo!("support #type_of for other types"),
                };
                let tag = self.builder.ins().iconst(types::I64, tag);

                let struct_slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size: struct_size,
                    offset: None,
                });
                let dest_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.module.isa().pointer_type(), struct_slot, 0);
                self.set_value(id, dest_addr);

                // store the tag
                self.builder.ins().store(MemFlags::new(), tag, dest_addr, 0);

                // store the data
                // self.builder
                //     .ins()
                //     .store(MemFlags::new(), global, dest_addr, 8);

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

    fn compile_call(&mut self, id: Id, name: Id, params: &Vec<Id>) -> Result<(), CompileError> {
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
        let return_size = type_size(&self.backend.semantic, &self.module, return_ty);

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
        let mut sig = self.module.make_signature();
        for param in params.iter() {
            sig.params.push(AbiParam::new(into_cranelift_type(
                &self.backend.semantic,
                *param,
            )?));
        }

        if return_size > 0 {
            sig.returns.push(AbiParam::new(into_cranelift_type(
                &self.backend.semantic,
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
}

pub fn into_cranelift_type(semantic: &Semantic, id: Id) -> Result<types::Type, CompileError> {
    let range = semantic.parser.ranges[id];
    let ty = &semantic.types[id];

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
        Type::Struct { params, .. } if semantic.parser.id_vec(*params).len() == 0 => unreachable!(),
        Type::Tokens => Ok(types::I64), // just the index into token_vecs
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
        Type::Tokens => 8,
        Type::Pointer(_) => module.isa().pointer_bytes() as _,
        Type::Func { .. } => module.isa().pointer_bytes() as _,
        Type::Struct { params, .. } => semantic
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
            "type_size for {:?} ({:?}) with node id {}",
            &semantic.types[id],
            // semantic.parser.debug(id),
            semantic.parser.nodes[id],
            id,
        ),
    }
}

fn declare_externs<'a, 'b>(
    semantic: &Semantic<'a>,
    module: &mut Module<SimpleJITBackend>,
    builder: &mut FunctionBuilder<'b>,
    values: &mut Vec<Value>,
) -> Result<(), CompileError> {
    for ext in semantic.parser.externs.clone() {
        let mut sig = module.make_signature();

        let (name, params, return_ty) = match semantic.parser.nodes[ext] {
            Node::Extern {
                name,
                params,
                return_ty,
            } => (name, params, return_ty),
            _ => unreachable!(),
        };

        let return_size = type_size(semantic, &module, return_ty);
        if return_size > 0 {
            sig.returns
                .push(AbiParam::new(into_cranelift_type(semantic, return_ty)?));
        }

        for param in semantic.parser.id_vec(params).clone() {
            sig.params
                .push(AbiParam::new(into_cranelift_type(semantic, param)?));
        }

        let name = semantic
            .parser
            .lexer
            .string_interner
            .resolve(name)
            .unwrap()
            .to_string();

        let func_id = module
            .declare_function(&name, Linkage::Import, &sig)
            .unwrap();

        values[ext] = Value::FuncRef(module.declare_func_in_func(func_id, &mut builder.func));
    }

    Ok(())
}

fn dynamic_fn_ptr(sym: Sym) -> *const u8 {
    FUNC_PTRS.lock().unwrap().get(&sym).unwrap().0
}

fn print_int(n: i64) {
    println!("{}", n);
}

fn print_string(len: i64, bytes: *mut u8) {
    let s = unsafe { String::from_raw_parts(bytes, len as usize, len as usize) };
    println!("{}", s);
    std::mem::forget(s);
}
