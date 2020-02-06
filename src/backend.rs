use crate::parser::{BasicType, CompileError, Id, Node, Parser, Type};
use crate::semantic::Semantic;

use cranelift::codegen::ir::{
    FuncRef, MemFlags, Signature, StackSlot, StackSlotData, StackSlotKind,
};
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
    StackSlot(StackSlot),
    // similar to a stackslot, but for general memory (or when the slot index isn't known, such as in a deref)
    Address(CraneliftValue),
}

impl Value {
    fn as_addr(&self) -> Value {
        *match self {
            Value::StackSlot(ss) => Some(self),
            Value::Address(addr) => Some(self),
            _ => None,
        }
        .unwrap()
    }

    fn as_func_ref(&self) -> Sym {
        *match self {
            Value::FuncRef(fr) => Some(fr),
            _ => None,
        }
        .unwrap()
    }

    fn as_value(&self) -> CraneliftValue {
        *match self {
            Value::Value(v) => Some(v),
            _ => None,
        }
        .unwrap()
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
        self.semantic.parser.calls.clear();

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

        let mut parser = Parser::new(&source);
        parser.parse()?;

        let mut semantic = Semantic::new(parser);
        semantic.assign_top_level_types()?;

        let mut backend = Backend::new(semantic);
        backend.compile()?;

        Ok(backend)
    }

    pub fn recompile_function(&mut self, function: impl Into<String>) -> Result<(), CompileError> {
        self.values.clear();
        while self.values.len() < self.semantic.parser.nodes.len() {
            self.values.push(Value::Unassigned);
        }

        let new_id = self.semantic.parser.get_top_level(function).unwrap();
        for topo in self.semantic.topo.clone() {
            if topo == new_id {
                continue;
            }

            match self.semantic.parser.nodes[topo] {
                Node::Func { name, .. } => {
                    // set the value to come from the FUNC_PTRS map
                    // Some(FUNC_PTRS.lock().unwrap().get(&name).unwrap())
                    // println!("inserting {} as funcref", topo);
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

                println!("compiling func {}", func_name);

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
                    local_id: 0,
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

                // println!("inserting {} as funcref", id);
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
    local_id: u32,
    dynamic_fn_ptr_decl: FuncRef,
    module: &'a mut Module<SimpleJITBackend>,
}

impl<'a, 'b, 'c> FunctionBackend<'a, 'b, 'c> {
    fn set_value(&mut self, id: Id, value: CraneliftValue) {
        self.values[id] = Value::Value(value);
    }

    fn next_local_id(&mut self) -> u32 {
        self.local_id += 1;
        self.local_id - 1
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
            _ => todo!("implement type_size for non-basic types"),
        }
    }

    fn store(&mut self, id: Id, dest: &Value) {
        match self.values[id] {
            Value::Value(_) => self.store_value(id, dest),
            _ => self.store_copy(id, dest),
        }
    }

    fn store_value(&mut self, id: Id, dest: &Value) {
        let source_value = self.values[id].as_value();
        match dest {
            Value::StackSlot(slot) => {
                self.builder.ins().stack_store(source_value, *slot, 0);
            }
            Value::Address(addr) => {
                self.builder
                    .ins()
                    .store(MemFlags::new(), source_value, *addr, 0);
            }
            _ => todo!("implement store_value"),
        }
    }

    fn store_copy(&mut self, id: Id, dest: &Value) {
        let size = self.type_size(id);

        let source_value = match &self.values[id] {
            Value::StackSlot(slot) => {
                self.builder
                    .ins()
                    .stack_addr(self.module.isa().pointer_type(), *slot, 0)
            }
            Value::Address(addr) => *addr,
            _ => todo!("implement store_copy for {:?}", &self.values[id]),
        };

        let dest_value = match dest {
            Value::StackSlot(slot) => {
                self.builder
                    .ins()
                    .stack_addr(self.module.isa().pointer_type(), *slot, 0)
            }
            Value::Address(addr) => *addr,
            _ => todo!("implement store_copy for {:?}", &self.values[id]),
        };

        self.builder.emit_small_memcpy(
            self.module.isa().frontend_config(),
            dest_value,
            source_value,
            size as _,
            1,
            1,
        );
    }

    fn load_value(&mut self, id: Id) -> CraneliftValue {
        self.load_value_with_type(id, id)
    }

    fn load_value_with_type(&mut self, id: Id, ty: Id) -> CraneliftValue {
        match &self.values[id] {
            Value::StackSlot(slot) => {
                let ty = &self.semantic.types[ty];

                self.builder
                    .ins()
                    .stack_load(ty.try_into().unwrap(), *slot, 0)
            }
            Value::Address(addr) => {
                let ty = &self.semantic.types[ty];

                self.builder
                    .ins()
                    .load(ty.try_into().unwrap(), MemFlags::new(), *addr, 0)
            }
            Value::Value(v) => *v,
            _ => panic!("Could not get cranelift value"),
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

                let value = self.load_value(*return_id);
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
                        _ => todo!("handle other type"),
                    },
                    _ => todo!("expected basic type"),
                };

                self.set_value(id, value);

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
            Node::Symbol(sym) => {
                let resolved = self.semantic.scope_get(*sym, id)?;
                self.compile_id(resolved)?;
                self.values[id] = self.values[resolved].clone();

                Ok(())
            }
            Node::Add(lhs, rhs) => {
                self.compile_id(*lhs)?;
                self.compile_id(*rhs)?;

                let lhs = self.load_value(*lhs);
                let rhs = self.load_value(*rhs);

                let value = self.builder.ins().iadd(lhs, rhs);
                self.set_value(id, value);

                Ok(())
            }
            Node::Let {
                name: _, // Sym
                ty,      // Id
                expr,    // Id
            } => {
                let size = self.type_size(*ty);
                let slot = self.builder.create_stack_slot(StackSlotData {
                    kind: StackSlotKind::ExplicitSlot,
                    size,
                    offset: None,
                });

                let slot = Value::StackSlot(slot);

                self.compile_id(*expr)?;
                self.store(*expr, &slot);

                self.values[id] = slot;

                Ok(())
            }
            Node::Set {
                name, // Id
                expr, // Id
            } => {
                self.compile_id(*name)?;
                let addr = self.values[*name].as_addr();
                self.compile_id(*expr)?;
                self.store(*expr, &addr);

                Ok(())
            }
            Node::Call {
                name,   // Id
                params, // Vec<Id>
                is_macro: _,
                is_indirect: _,
            } => {
                // todo(chad): This doesn't need to be dynamic (call and then call-indirect) when compiling straight to an object file

                self.compile_id(*name)?;
                for param in params {
                    self.compile_id(*param)?;
                }

                let func_sym = self.values[*name].as_func_ref();

                let cranelift_params = [self
                    .builder
                    .ins()
                    .iconst(types::I64, func_sym.to_usize() as i64)];

                let call_inst = self
                    .builder
                    .ins()
                    .call(self.dynamic_fn_ptr_decl, &cranelift_params);
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
                    .map(|param| self.load_value(*param))
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
                self.values[id] = Value::StackSlot(slot);

                let true_block = self.builder.create_ebb();
                let false_block = self.builder.create_ebb();
                let cont_block = self.builder.create_ebb();

                self.compile_id(*cond_id)?;
                let cond = self.load_value(*cond_id);

                self.builder.ins().brnz(cond, true_block, &[]);
                self.builder.ins().jump(false_block, &[]);

                // true
                {
                    self.builder.switch_to_block(true_block);
                    self.current_block = true_block;
                    self.compile_id(*true_id)?;

                    self.store(*true_id, &Value::StackSlot(slot));

                    self.builder.ins().jump(cont_block, &[]);
                }

                // false
                if false_id.is_some() {
                    self.builder.switch_to_block(false_block);
                    self.current_block = false_block;
                    self.compile_id(false_id.unwrap())?;

                    self.store(false_id.unwrap(), &Value::StackSlot(slot));

                    self.builder.ins().jump(cont_block, &[]);
                }

                self.builder.switch_to_block(cont_block);
                self.current_block = cont_block;

                Ok(())
            }
            Node::Ref(ref_id) => {
                self.compile_id(*ref_id)?;
                let slot = match self.values[*ref_id] {
                    Value::StackSlot(ss) => Some(ss),
                    _ => None,
                }
                .unwrap();

                self.values[id] = Value::Value(self.builder.ins().stack_addr(
                    self.module.isa().pointer_type(),
                    slot,
                    0,
                ));

                Ok(())
            }
            Node::Load(load_id) => {
                self.compile_id(*load_id)?;
                let addr = self.load_value_with_type(*load_id, id);
                self.values[id] = Value::Address(addr);

                Ok(())
            }
            Node::FuncParam {
                name: _, // Sym
                ty: _,   // Id
                index,   // u16
            } => {
                let params = self.builder.ebb_params(self.current_block);
                self.values[id] = Value::Value(params[*index as usize]);
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
}

impl TryInto<types::Type> for &Type {
    type Error = &'static str;

    fn try_into(self) -> Result<types::Type, &'static str> {
        match self {
            &Type::Basic(bt) => match bt {
                BasicType::Bool => Ok(types::I32), // todo(chad): I8?
                BasicType::I8 => Ok(types::I8),
                BasicType::I16 => Ok(types::I16),
                BasicType::I32 => Ok(types::I32),
                BasicType::I64 => Ok(types::I64),
                BasicType::F32 => Ok(types::F32),
                BasicType::F64 => Ok(types::F64),
                _ => Err("Could not convert type"),
            },
            &Type::Pointer(_) => Ok(types::I64), // todo(chad): need to get the actual type here, from the isa
            _ => Err("Could not convert type"),
        }
    }
}

// todo(chad): this would be dynamically generated/cached on the fly for the real version, so we could handle any combination of arguments
fn dynamic_fn_ptr(sym: Sym) -> *const u8 {
    FUNC_PTRS.lock().unwrap().get(&sym).unwrap().0
}
