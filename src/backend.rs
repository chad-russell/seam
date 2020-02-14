use crate::parser::{BasicType, CompileError, Id, Node, Parser, Type};
use crate::semantic::Semantic;

use cranelift::codegen::ir::{
    FuncRef, MemFlags, Signature, StackSlot, StackSlotData, StackSlotKind,
};
use cranelift::codegen::settings::{self, Configurable};
use cranelift::prelude::{
    types, AbiParam, Ebb, ExternalName, FunctionBuilder, FunctionBuilderContext, InstBuilder,
    Value as CraneliftValue,
};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};

use std::collections::BTreeMap;
use std::convert::TryInto;
use std::ops::Deref;
use std::ops::DerefMut;

use std::sync::{Arc, Mutex};

use string_interner::Symbol;

type Sym = usize;

#[derive(Clone, Copy, Debug)]
pub struct FuncPtr(pub *const u8);

unsafe impl Send for FuncPtr {}
unsafe impl Sync for FuncPtr {}

const ENUM_TAG_SIZE_BYTES: u32 = 2;

// todo(chad): replace with evmap or similar?
lazy_static! {
    pub static ref FUNC_PTRS: Arc<Mutex<BTreeMap<Sym, FuncPtr>>> =
        Arc::new(Mutex::new(BTreeMap::new()));
}

pub struct Builder<'a> {
    builder: FunctionBuilder<'a>,
}

impl<'a> Deref for Builder<'a> {
    type Target = FunctionBuilder<'a>;

    fn deref(&self) -> &Self::Target {
        &self.builder
    }
}

impl<'a> DerefMut for Builder<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.builder
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Value {
    Unassigned,
    None,
    FuncRef(Sym),
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
        match self {
            Value::Value(_) => Some(self),
            _ => None,
        }
        .unwrap()
        .clone()
    }

    fn as_value_relaxed<'a, 'b, 'c>(
        &self,
        backend: &mut FunctionBackend<'a, 'b, 'c>,
    ) -> CraneliftValue {
        match self {
            Value::Value(v) => *v,
            Value::FuncRef(sym) => backend.builder.ins().iconst(types::I64, *sym as i64),
            _ => todo!("as_value_relaxed for {:?}", self),
        }
    }
}

pub struct Backend<'a> {
    pub semantic: Semantic<'a>,
    func_ctx: FunctionBuilderContext,
    pub values: Vec<Value>,
    funcs: BTreeMap<Sym, FuncId>,
    func_sigs: BTreeMap<Sym, Signature>,
}

impl<'a> Backend<'a> {
    pub fn new(semantic: Semantic<'a>) -> Self {
        let func_ctx = FunctionBuilderContext::new();
        let len = semantic.types.len();

        Self {
            semantic,
            func_ctx,
            values: vec![Value::Unassigned; len],
            funcs: BTreeMap::new(),
            func_sigs: BTreeMap::new(),
        }
    }

    pub fn update_source(&mut self, source: &'a str) -> Result<(), CompileError> {
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
        self.semantic.assign_top_level_types()?;

        Ok(())
    }

    pub fn bootstrap_from_source(source: &str) -> Result<Backend, CompileError> {
        FUNC_PTRS.lock().unwrap().clear();

        let now = std::time::Instant::now();
        let mut parser = Parser::new(&source);
        parser.parse()?;
        let elapsed = now.elapsed();
        println!(
            "parser ran in: {}",
            elapsed.as_micros() as f64 / 1_000_000.0
        );

        let mut semantic = Semantic::new(parser);
        semantic.assign_top_level_types()?;
        let elapsed = now.elapsed();
        println!(
            "parser + semantic ran in: {}",
            elapsed.as_micros() as f64 / 1_000_000.0
        );

        let mut backend = Backend::new(semantic);
        backend.compile()?;
        println!(
            "parser + semantic + backend ran in: {}",
            elapsed.as_micros() as f64 / 1_000_000.0
        );

        Ok(backend)
    }

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
                    self.values[topo] = Value::FuncRef(name);
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

    pub fn call_func(&self, str: &str) -> i32 {
        let f: fn() -> i32 = unsafe {
            std::mem::transmute::<_, fn() -> i32>(
                (FUNC_PTRS.lock().unwrap())
                    [&self.semantic.parser.lexer.string_interner.get(str).unwrap()],
            )
        };
        f()
    }

    pub fn compile(&mut self) -> Result<(), CompileError> {
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

        match &self.semantic.parser.nodes[id] {
            Node::Func {
                name,      // Sym,
                params,    // Vec<Id>,
                return_ty, // Id,
                stmts,     // Vec<Id>,
                ..
            } => {
                // todo(chad): find a way to reuse these
                let mut module: Module<SimpleJITBackend> = {
                    let mut jit_builder = SimpleJITBuilder::new(default_libcall_names());
                    jit_builder.symbol("__dynamic_fn_ptr", dynamic_fn_ptr as *const u8);
                    for (&name, &ptr) in FUNC_PTRS.lock().unwrap().iter() {
                        jit_builder.symbol(self.resolve_symbol(name), ptr.0);
                    }

                    Module::new(jit_builder)
                };

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

                let func_name = String::from(self.semantic.parser.resolve_sym_unchecked(*name));
                let func_sym = self.get_symbol(func_name.clone());

                // println!("compiling func {}", func_name);

                let mut sig = module.make_signature();

                for param in params {
                    sig.params
                        .push(AbiParam::new(self.get_cranelift_type(*param)?));
                }
                sig.returns
                    .push(AbiParam::new(self.get_cranelift_type(*return_ty)?));
                ctx.func.signature = sig;

                self.func_sigs.insert(func_sym, ctx.func.signature.clone());

                let func = module
                    .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
                    .unwrap();
                ctx.func.name = ExternalName::user(0, func.as_u32());

                let mut builder = Builder {
                    builder: FunctionBuilder::new(&mut ctx.func, &mut self.func_ctx),
                };
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

                let mut fb = FunctionBackend {
                    semantic: &self.semantic,
                    values: &mut self.values,
                    builder: &mut builder,
                    current_block: ebb,
                    dynamic_fn_ptr_decl: dfp_decl,
                    module: &mut module,
                };
                for stmt in stmts.iter() {
                    fb.compile_id(*stmt)?;
                }

                builder.seal_all_blocks();
                builder.finalize();

                // println!("{}", ctx.func.display(None));

                module.define_function(func, &mut ctx).unwrap();
                module.clear_context(&mut ctx);

                self.values[id] = Value::FuncRef(func_sym);

                module.finalize_definitions();

                let func = module.get_finalized_function(func);
                FUNC_PTRS.lock().unwrap().insert(func_sym, FuncPtr(func));

                Ok(())
            }
            Node::TypeLiteral(_) => {
                self.values[id] = Value::None;
                Ok(())
            }
            _ => Err(CompileError::from_string(
                format!("Unhandled node: {}", self.semantic.parser.debug(id)),
                self.semantic.parser.ranges[id],
            )),
        }
    }

    fn get_cranelift_type(&self, id: Id) -> Result<types::Type, CompileError> {
        get_cranelift_type(&self.semantic, id)
    }
}

fn get_cranelift_type(semantic: &Semantic, id: Id) -> Result<types::Type, CompileError> {
    let range = semantic.parser.ranges[id];

    semantic
        .types
        .get(id)
        .ok_or(CompileError::from_string("Type not found", range))?
        .try_into()
        .map_err(|e| CompileError::from_string(String::from(e), range))
}

struct FunctionBackend<'a, 'b, 'c> {
    semantic: &'a Semantic<'b>,
    values: &'a mut Vec<Value>,
    builder: &'a mut Builder<'c>,
    current_block: Ebb,
    dynamic_fn_ptr_decl: FuncRef,
    module: &'a mut Module<SimpleJITBackend>,
}

impl<'a, 'b, 'c> FunctionBackend<'a, 'b, 'c> {
    fn set_value(&mut self, id: Id, value: CraneliftValue) {
        self.values[id] = Value::Value(value);
    }

    fn type_size(&self, id: Id) -> u32 {
        match &self.semantic.types[id] {
            Type::Basic(BasicType::None) => 0,
            Type::Basic(BasicType::Bool) => 4,
            Type::Basic(BasicType::I8) => 1,
            Type::Basic(BasicType::I16) => 2,
            Type::Basic(BasicType::I32) => 4,
            Type::Basic(BasicType::I64) => 8,
            Type::Basic(BasicType::F32) => 4,
            Type::Basic(BasicType::F64) => 8,
            Type::Pointer(_) => self.module.isa().pointer_bytes() as _,
            Type::Func { .. } => self.module.isa().pointer_bytes() as _,
            Type::Struct(params) => params.iter().map(|p| self.type_size(*p)).sum(),
            Type::Enum(params) => {
                let biggest_param = params.iter().map(|p| self.type_size(*p)).max().unwrap_or(0);
                let tag_size = ENUM_TAG_SIZE_BYTES;
                tag_size + biggest_param
            }
            _ => todo!(
                "type_size for {:?} ({})",
                &self.semantic.types[id],
                self.semantic.parser.debug(id)
            ),
        }
    }

    // Returns whether the value for a particular id is the value itself, or a pointer to the value
    // For instance, the constant integer 3 would likely be just the value, and not a pointer.
    // However a field access (or even a load) needs to always be returned by pointer, because it might refer to a struct literal
    // Some nodes (like const int above) don't usually return a pointer, but they might need to for other reasons,
    // e.g. they need their address to be taken. In that case we store that fact in the 'node_has_slot' vec.
    // Also symbols won't know whether they have a local or not because it depends what they're referencing. So 'node_has_slot' is handy there too.
    fn rvalue_is_ptr(&mut self, id: Id) -> bool {
        if self.semantic.parser.node_is_addressable[id] {
            return true;
        }

        match &self.semantic.parser.nodes[id] {
            Node::Load(load_id) => self.rvalue_is_ptr(*load_id),
            Node::ValueParam { value, .. } => self.rvalue_is_ptr(*value),
            Node::Field { .. } => true,
            Node::Let { .. } => true,
            _ => false,
        }
    }

    fn as_value(&mut self, id: Id) -> CraneliftValue {
        match self.values[id].clone() {
            Value::Value(v) => Some(v),
            Value::FuncRef(sym) => Some(self.builder.ins().iconst(types::I64, sym as i64)),
            _ => None,
        }
        .expect(&format!(
            "Cannot convert {:?} into a CraneliftValue",
            &self.values[id]
        ))
    }

    fn store(&mut self, id: Id, dest: &Value) {
        if self.rvalue_is_ptr(id) {
            self.store_copy(id, dest);
        } else {
            self.store_value(id, dest);
        }
    }

    fn store_value(&mut self, id: Id, dest: &Value) {
        let source_value = self.values[id].clone().as_value_relaxed(self);

        match dest {
            Value::Value(value) => {
                self.builder
                    .ins()
                    .store(MemFlags::new(), source_value, *value, 0);
            }
            _ => todo!("store_value dest for {:?}", dest),
        }
    }

    fn store_copy(&mut self, id: Id, dest: &Value) {
        let size = self.type_size(id);

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
        let rvalue_is_ptr = self.rvalue_is_ptr(id);

        if rvalue_is_ptr {
            let ty = &self.semantic.types[id];
            let ty = match ty {
                Type::Struct(_) => self.module.isa().pointer_type(),
                Type::Enum(_) => self.module.isa().pointer_type(),
                _ => ty.try_into().unwrap(),
            };

            match &self.values[id] {
                Value::Value(v) => self.builder.ins().load(ty, MemFlags::new(), *v, 0),
                _ => panic!("Could not get cranelift value: {:?}", &self.values[id]),
            }
        } else {
            self.as_value(id)
        }
    }

    fn compile_id(&mut self, id: Id) -> Result<(), CompileError> {
        // idempotency
        match &self.values[id] {
            Value::Unassigned => {}
            _ => return Ok(()),
        };

        match &self.semantic.parser.nodes[id] {
            Node::Return(return_id) => {
                self.compile_id(*return_id)?;

                let value = self.rvalue(*return_id);
                self.values[id] = Value::Value(value);

                self.builder.ins().return_(&[value]);

                Ok(())
            }
            Node::IntLiteral(i) => {
                let value = match self.semantic.types[id] {
                    Type::Basic(bt) => match bt {
                        BasicType::I64 => self.builder.ins().iconst(types::I64, *i),
                        BasicType::I32 => self.builder.ins().iconst(types::I32, *i),
                        BasicType::I16 => self.builder.ins().iconst(types::I16, *i),
                        BasicType::I8 => self.builder.ins().iconst(types::I8, *i),
                        _ => todo!(
                            "handle other type: {:?} {:?}",
                            &self.semantic.types[id],
                            &self.semantic.parser.nodes[id]
                        ),
                    },
                    _ => todo!("expected basic type"),
                };

                self.set_value(id, value);

                if self.semantic.parser.node_is_addressable[id] {
                    let size = self.type_size(id);
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
                    self.store_value(id, &value);
                    self.values[id] = value;
                }

                Ok(())
            }
            Node::BoolLiteral(b) => {
                let value = self
                    .builder
                    .ins()
                    .iconst(types::I32, if *b { 1 } else { 0 });
                self.set_value(id, value);

                Ok(())
            }
            Node::StructLiteral { name: _, params } => {
                // if this typechecked as an enum, handle things a bit differently
                match &self.semantic.types[id] {
                    Type::Enum(_) => {
                        assert_eq!(params.len(), 1);
                        let param = params[0];
                        let size = self.type_size(param) + ENUM_TAG_SIZE_BYTES;

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

                        self.values[id] = Value::Value(addr);

                        let index = match &self.semantic.parser.nodes[param] {
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
                let size = self.type_size(id);
                let slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let mut addr =
                    self.builder
                        .ins()
                        .stack_addr(self.module.isa().pointer_type(), slot, 0);

                self.values[id] = Value::Value(addr);

                for param in params {
                    self.compile_id(*param)?;
                    self.store(*param, &Value::Value(addr));

                    let param_size = self.type_size(*param) as i64;
                    let param_size = self.builder.ins().iconst(types::I64, param_size);
                    addr = self.builder.ins().iadd(addr, param_size);
                }

                Ok(())
            }
            Node::Symbol(sym) => {
                let resolved = self.semantic.scope_get(*sym, id)?;
                self.compile_id(resolved)?;
                self.values[id] = self.values[resolved].clone();

                Ok(())
            }
            Node::Add(lhs, rhs) => {
                self.compile_id(*lhs)?;
                self.compile_id(*rhs)?;

                let lhs = self.rvalue(*lhs);
                let rhs = self.rvalue(*rhs);

                let value = self.builder.ins().iadd(lhs, rhs);
                self.set_value(id, value);

                Ok(())
            }
            Node::Let {
                name: _, // Sym
                ty: _,   // Id
                expr,    // Id
            } => {
                let size = self.type_size(id);
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
                    self.compile_id(*expr)?;
                    self.store(*expr, &value);
                }

                self.values[id] = value;

                Ok(())
            }
            Node::Set {
                name,     // Id
                expr,     // Id
                is_store, // bool
            } => {
                if *is_store {
                    // Don't actually codegen the load, just codegen what it is loading so we have the pointer. Then it's easy to store into
                    let name = match &self.semantic.parser.nodes[*name] {
                        Node::Load(load_id) => *load_id,
                        _ => unreachable!(),
                    };

                    self.compile_id(name)?;
                    self.compile_id(*expr)?;

                    let mut addr = self.values[name].clone().as_value_relaxed(self);
                    if self.rvalue_is_ptr(name) {
                        addr = self.builder.ins().load(
                            self.module.isa().pointer_type(),
                            MemFlags::new(),
                            addr,
                            0,
                        );
                    }

                    self.store(*expr, &Value::Value(addr));
                } else {
                    self.compile_id(*name)?;

                    let addr = self.values[*name].as_addr();

                    self.compile_id(*expr)?;
                    self.store(*expr, &addr);
                }

                Ok(())
            }
            Node::Call {
                name,      // Id
                ct_params, // Vec<Id>
                params,    // Vec<Id>
                is_indirect: _,
            } => {
                // todo(chad): This doesn't need to be dynamic (call and then call-indirect) when compiling straight to an object file

                self.compile_id(*name)?;
                for param in params {
                    self.compile_id(*param)?;
                }

                let cranelift_param = match &self.values[*name] {
                    Value::FuncRef(fr) => {
                        self.builder.ins().iconst(types::I64, fr.to_usize() as i64)
                    }
                    Value::Value(_) => self.rvalue(*name),
                    _ => unreachable!("unrecognized Value type in codegen Node::Call"),
                };

                let call_inst = self
                    .builder
                    .ins()
                    .call(self.dynamic_fn_ptr_decl, &[cranelift_param]);
                let value = self.builder.func.dfg.inst_results(call_inst)[0];

                // Now call indirect
                let func_ty = &self.semantic.types[*name];
                let return_ty = match func_ty {
                    Type::Func { return_ty, .. } => Some(*return_ty),
                    _ => None,
                }
                .unwrap();

                let mut sig = self.module.make_signature();
                for param in params.iter() {
                    sig.params
                        .push(AbiParam::new(get_cranelift_type(&self.semantic, *param)?));
                }

                sig.returns.push(AbiParam::new(get_cranelift_type(
                    &self.semantic,
                    return_ty,
                )?));

                let cranelift_params = params
                    .iter()
                    .map(|param| self.rvalue(*param))
                    .collect::<Vec<_>>();

                let sig = self.builder.import_signature(sig);
                let call_inst = self
                    .builder
                    .ins()
                    .call_indirect(sig, value, &cranelift_params);

                let value = self.builder.func.dfg.inst_results(call_inst)[0];
                self.set_value(id, value);

                Ok(())
            }
            Node::If(cond_id, true_id, false_id) => {
                let size = self.type_size(id);
                let slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });
                let slot_addr =
                    self.builder
                        .ins()
                        .stack_addr(self.module.isa().pointer_type(), slot, 0);
                self.values[id] = Value::Value(slot_addr);

                let true_block = self.builder.create_ebb();
                let false_block = self.builder.create_ebb();
                let cont_block = self.builder.create_ebb();

                self.compile_id(*cond_id)?;
                let cond = self.rvalue(*cond_id);

                self.builder.ins().brnz(cond, true_block, &[]);
                self.builder.ins().jump(false_block, &[]);

                // true
                {
                    self.builder.switch_to_block(true_block);
                    self.current_block = true_block;
                    self.compile_id(*true_id)?;

                    self.store(*true_id, &Value::Value(slot_addr));

                    self.builder.ins().jump(cont_block, &[]);
                }

                // false
                if false_id.is_some() {
                    self.builder.switch_to_block(false_block);
                    self.current_block = false_block;
                    self.compile_id(false_id.unwrap())?;

                    self.store(false_id.unwrap(), &Value::Value(slot_addr));

                    self.builder.ins().jump(cont_block, &[]);
                }

                self.builder.switch_to_block(cont_block);
                self.current_block = cont_block;

                Ok(())
            }
            Node::Ref(ref_id) => {
                self.compile_id(*ref_id)?;
                match self.values[*ref_id] {
                    Value::Value(_) => self.values[id] = self.values[*ref_id],
                    _ => todo!("{:?}", &self.values[*ref_id]),
                };

                // if we are meant to be addressable, then we need to store our actual value into something which has an address
                if self.semantic.parser.node_is_addressable[id] {
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
                    self.store_value(id, &value);
                    self.values[id] = value;
                }

                Ok(())
            }
            Node::Load(load_id) => {
                self.compile_id(*load_id)?;

                let value = match &self.values[*load_id] {
                    Value::Value(value) => *value,
                    _ => todo!("load for {:?}", &self.values[*load_id]),
                };

                let ty = &self.semantic.types[id];
                let ty = match ty {
                    Type::Struct(_) => self.module.isa().pointer_type(),
                    _ => ty.try_into().unwrap(),
                };

                let loaded = self.builder.ins().load(ty, MemFlags::new(), value, 0);

                self.set_value(id, loaded);

                Ok(())
            }
            Node::DeclParam {
                name: _, // Sym
                ty: _,   // Id
                index,   // u16
            } => {
                let params = self.builder.ebb_params(self.current_block);
                self.values[id] = Value::Value(params[*index as usize]);
                Ok(())
            }
            Node::ValueParam {
                name: _, // Sym
                value,
                ..
            } => {
                self.compile_id(*value)?;
                self.values[id] = self.values[*value];
                Ok(())
            }
            Node::Field {
                base,
                field_name: _,
                field_index,
                is_assignment,
            } => {
                self.compile_id(*base)?;

                // todo(chad): deref more than once?
                let (unpointered_ty, loaded) = match &self.semantic.types[*base] {
                    Type::Pointer(id) => (*id, true),
                    _ => (*base, false),
                };

                // all field accesses on enums have the same offset -- just the tag size in bytes
                let is_enum = match &self.semantic.types[unpointered_ty] {
                    Type::Enum(_) => true,
                    _ => false,
                };

                let offset = if is_enum {
                    ENUM_TAG_SIZE_BYTES
                } else {
                    self.semantic.types[unpointered_ty]
                        .as_struct_params()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(_index, id)| {
                            let (_name, ty, _index) =
                                self.semantic.parser.nodes[*id].as_param().unwrap();
                            ty
                        })
                        .take(*field_index as usize)
                        .map(|ty| self.type_size(ty))
                        .sum::<u32>()
                };

                let mut base = self.as_value(*base);

                if loaded {
                    // if we are doing field access through a pointer, do an extra load
                    base = self.builder.ins().load(
                        self.module.isa().pointer_type(),
                        MemFlags::new(),
                        base,
                        0,
                    );
                }

                if is_enum && *is_assignment {
                    let tag_value = self.builder.ins().iconst(types::I16, *field_index as i64);
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

                self.values[id] = base.into();

                Ok(())
            }
            Node::TypeLiteral(_) => {
                self.values[id] = Value::None;
                Ok(())
            }
            _ => Err(CompileError::from_string(
                format!(
                    "Unhandled node: {:?} ({})",
                    &self.semantic.parser.nodes[id],
                    self.semantic.parser.debug(id)
                ),
                self.semantic.parser.ranges[id],
            )),
        }
    }
}

impl TryInto<types::Type> for &Type {
    type Error = String;

    fn try_into(self) -> Result<types::Type, String> {
        match self {
            &Type::Basic(bt) => match bt {
                BasicType::Bool => Ok(types::I32), // todo(chad): I8?
                BasicType::I8 => Ok(types::I8),
                BasicType::I16 => Ok(types::I16),
                BasicType::I32 => Ok(types::I32),
                BasicType::I64 => Ok(types::I64),
                BasicType::F32 => Ok(types::F32),
                BasicType::F64 => Ok(types::F64),
                _ => Err(format!("Could not convert type {:?}", &self)),
            },
            &Type::Func { .. } => Ok(types::I64),
            &Type::Pointer(_) => Ok(types::I64), // todo(chad): need to get the actual type here, from the isa
            _ => Err(format!("Could not convert type {:?}", &self)),
        }
    }
}

// todo(chad): this would be dynamically generated/cached on the fly for the real version, so we could handle any combination of arguments
fn dynamic_fn_ptr(sym: Sym) -> *const u8 {
    FUNC_PTRS.lock().unwrap().get(&sym).unwrap().0
}
