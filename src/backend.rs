use crate::parser::{BasicType, CompileError, Id, Node, Parser, Type};
use crate::semantic::Semantic;

use cranelift::codegen::ir::{FuncRef, Signature};
use cranelift::prelude::{
    types, AbiParam, ExternalName, FunctionBuilder, FunctionBuilderContext, InstBuilder,
    Value as CraneliftValue, Variable,
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
    Variable(Variable),
}

impl Value {
    fn as_cranelift_value(&self) -> CraneliftValue {
        *match self {
            Value::Value(cv) => Some(cv),
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
                    jit_builder.symbol("__wrapper", wrapper as *const u8);
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

                let wrapper_func = {
                    let mut wrapper_sig = module.make_signature();

                    wrapper_sig.params.push(AbiParam::new(types::I32));
                    wrapper_sig.returns.push(AbiParam::new(types::I64));

                    let wrapper_func_id = module
                        .declare_function("__wrapper", Linkage::Import, &wrapper_sig)
                        .unwrap();

                    module.declare_func_in_func(wrapper_func_id, &mut builder.func)
                };

                let local_scope = self.semantic.parser.node_scopes[id];
                let local_scope = &self.semantic.parser.scopes[local_scope];

                let mut fb = FunctionBackend {
                    semantic: &self.semantic,
                    values: &mut self.values,
                    builder: &mut builder,
                    local_id: 0,
                    wrapper_func,
                };
                for stmt in stmts.iter() {
                    fb.compile_id(*stmt)?;
                }

                builder.seal_all_blocks();
                builder.finalize();

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
    local_id: u32,
    wrapper_func: FuncRef,
}

impl<'a, 'b, 'c> FunctionBackend<'a, 'b, 'c> {
    fn set_value(&mut self, id: Id, value: CraneliftValue) {
        self.values[id] = Value::Value(value);
    }

    fn next_local_id(&mut self) -> u32 {
        self.local_id += 1;
        self.local_id - 1
    }

    fn resolve_value(&mut self, id: Id) -> Value {
        match &self.values[id].clone() {
            Value::Variable(var_id) => Value::Value(self.builder.use_var(*var_id)),
            _ => self.values[id].clone(),
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
                let value = self.values[*return_id].clone();
                self.builder.ins().return_(&[value.as_cranelift_value()]);
                self.values[id] = value;

                Ok(())
            }
            Node::IntLiteral(i) => {
                let value = self.builder.ins().iconst(types::I64, *i);
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
                self.values[id] = self.resolve_value(resolved);

                Ok(())
            }
            Node::Add(lhs, rhs) => {
                self.compile_id(*lhs)?;
                self.compile_id(*rhs)?;

                let value = self.builder.ins().iadd(
                    self.values[*lhs].as_cranelift_value(),
                    self.values[*rhs].as_cranelift_value(),
                );
                self.set_value(id, value);

                Ok(())
            }
            Node::Let {
                name: _, // Sym
                ty: _,   // Id
                expr,    // Id
            } => {
                let var = Variable::with_u32(self.next_local_id());

                self.builder
                    .declare_var(var, get_cranelift_type(&self.semantic, id)?);
                self.values[id] = Value::Variable(var);

                self.compile_id(*expr)?;
                let val = self.resolve_value(*expr).as_cranelift_value();
                self.builder.def_var(var, val);

                Ok(())
            }
            Node::Call {
                name, // Id
                args, // Vec<Id>
                is_macro: _,
                is_indirect: _,
            } => {
                self.compile_id(*name)?;
                for arg in args {
                    self.compile_id(*arg)?;
                }

                // let cranelift_args = args
                //     .iter()
                //     .map(|a| self.values[*a].as_cranelift_value())
                //     .collect::<Vec<_>>();

                // println!("retrieving {} as func ref", *name);
                let func_sym = self.values[*name].as_func_ref();
                // let func_id = self.funcs[&func_ref];
                // let func_ref = self
                //     .module
                //     .declare_func_in_func(func_id, &mut self.builder.func);

                let cranelift_args = [self
                    .builder
                    .ins()
                    .iconst(types::I32, func_sym.to_usize() as i64)];

                let call_inst = self.builder.ins().call(self.wrapper_func, &cranelift_args);
                let value = self.builder.func.dfg.inst_results(call_inst)[0];
                self.set_value(id, value);

                Ok(())
            }
            Node::If(cond_id, true_id, false_id) => {
                let var = Variable::with_u32(self.next_local_id());

                self.builder
                    .declare_var(var, get_cranelift_type(&self.semantic, id)?);
                self.values[id] = Value::Variable(var);

                let true_block = self.builder.create_ebb();
                let false_block = self.builder.create_ebb();
                let cont_block = self.builder.create_ebb();

                self.compile_id(*cond_id)?;
                self.builder.ins().brnz(
                    self.values[*cond_id].as_cranelift_value(),
                    true_block,
                    &[],
                );
                self.builder.ins().jump(false_block, &[]);

                // true
                {
                    self.builder.switch_to_block(true_block);
                    self.compile_id(*true_id)?;

                    let resolved = self.resolve_value(*true_id).as_cranelift_value();
                    self.builder.def_var(var, resolved);

                    self.builder.ins().jump(cont_block, &[]);
                }

                // false
                if false_id.is_some() {
                    self.builder.switch_to_block(false_block);
                    self.compile_id(false_id.unwrap())?;

                    let resolved = self.resolve_value(false_id.unwrap()).as_cranelift_value();
                    self.builder.def_var(var, resolved);

                    self.builder.ins().jump(cont_block, &[]);
                }

                self.builder.switch_to_block(cont_block);

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
                BasicType::I32 => Ok(types::I32),
                BasicType::I64 => Ok(types::I64),
                BasicType::F32 => Ok(types::F32),
                BasicType::F64 => Ok(types::F64),
                _ => Err("Could not convert type"),
            },
            _ => Err("Could not convert type"),
        }
    }
}

// todo(chad): this would be dynamically generated/cached on the fly for the real version, so we could handle any combination of arguments
fn wrapper(sym: Sym) -> i64 {
    let fp = {
        let func_ptrs = FUNC_PTRS.lock().unwrap();
        func_ptrs[&sym]
    };

    let f = unsafe { std::mem::transmute::<_, fn() -> i64>(fp) };
    f()
}
