use crate::parser::{BasicType, CompileError, Id, Node, Type};
use crate::semantic::Semantic;

use cranelift::codegen::ir::Signature;
use cranelift::prelude::{
    types, AbiParam, ExternalName, FunctionBuilder, FunctionBuilderContext, InstBuilder,
    Value as CraneliftValue,
};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};

use std::collections::HashMap;
use std::convert::TryInto;
use std::ops::Deref;
use std::ops::DerefMut;

use std::sync::{Arc, Mutex};

use string_interner::{StringInterner, Sym};

#[derive(Clone, Copy)]
struct FuncPtr(*const u8);

unsafe impl Send for FuncPtr {}
unsafe impl Sync for FuncPtr {}

// todo(chad): replace with evmap or similar?
lazy_static! {
    static ref FUNC_PTRS: Arc<Mutex<HashMap<Sym, FuncPtr>>> = Arc::new(Mutex::new(HashMap::new()));
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
enum Value {
    Unassigned,
    None,
    FuncRef(Sym),
    Value(CraneliftValue),
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
    semantic: Semantic<'a>,
    func_ctx: FunctionBuilderContext,
    values: Vec<Value>,
    funcs: HashMap<Sym, FuncId>,
    func_sigs: HashMap<Sym, Signature>,
    pub string_interner: StringInterner<Sym>,
}

impl<'a> Backend<'a> {
    pub fn new(semantic: Semantic<'a>) -> Self {
        let func_ctx = FunctionBuilderContext::new();
        let len = semantic.types.len();

        Self {
            semantic,
            func_ctx,
            values: vec![Value::Unassigned; len],
            funcs: HashMap::new(),
            func_sigs: HashMap::new(),
            string_interner: StringInterner::new(),
        }
    }

    fn resolve_symbol(&self, sym: Sym) -> String {
        self.string_interner.resolve(sym).unwrap().to_string()
    }

    pub fn call_func_no_args_i32_return(&self, str: &str) -> i32 {
        let f: fn() -> i32 = unsafe {
            std::mem::transmute::<_, fn() -> i32>(
                (FUNC_PTRS.lock().unwrap())[&self.string_interner.get(str).unwrap()],
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

    fn compile_id(&mut self, id: Id) -> Result<(), CompileError> {
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
                let func_sym = self.string_interner.get_or_intern(&func_name);

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

                let mut fb = FunctionBackend {
                    module: &module,
                    semantic: &self.semantic,
                    values: &mut self.values,
                    builder: &mut builder,
                    funcs: &self.funcs,
                };
                for stmt in stmts.iter() {
                    fb.compile_id(*stmt)?;
                }

                builder.seal_all_blocks();
                builder.finalize();

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
                format!("Unhandled node: {:?}", &self.semantic.parser.nodes[id]),
                self.semantic.parser.ranges[id],
            )),
        }
    }

    fn get_cranelift_type(&self, id: Id) -> Result<types::Type, CompileError> {
        let range = self.semantic.parser.ranges[id];

        self.semantic
            .types
            .get(id)
            .ok_or(CompileError::from_string("Type not found", range))?
            .try_into()
            .map_err(|e| CompileError::from_string(String::from(e), range))
    }
}

struct FunctionBackend<'a, 'b, 'c> {
    module: &'a Module<SimpleJITBackend>,
    semantic: &'a Semantic<'b>,
    values: &'a mut Vec<Value>,
    builder: &'a mut Builder<'c>,
    funcs: &'a HashMap<Sym, FuncId>,
}

impl<'a, 'b, 'c> FunctionBackend<'a, 'b, 'c> {
    fn set_value(&mut self, id: Id, value: CraneliftValue) {
        self.values[id] = Value::Value(value);
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
            Node::Symbol(sym) => {
                let resolved = self.semantic.scope_get(*sym, id)?;
                self.compile_id(resolved)?;
                self.values[id] = self.values[resolved].clone();
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

                let cranelift_args = args
                    .iter()
                    .map(|a| self.values[*a].as_cranelift_value())
                    .collect::<Vec<_>>();

                let func_ref = self.values[*name].as_func_ref();
                let func_id = self.funcs[&func_ref];
                let func_ref = self
                    .module
                    .declare_func_in_func(func_id, &mut self.builder.func);

                let call_inst = self.builder.ins().call(func_ref, &cranelift_args);
                let value = self.builder.func.dfg.inst_results(call_inst)[0];
                self.set_value(id, value);

                Ok(())
            }
            Node::TypeLiteral(_) => {
                self.values[id] = Value::None;
                Ok(())
            }
            _ => Err(CompileError::from_string(
                format!("Unhandled node: {:?}", &self.semantic.parser.nodes[id]),
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
    println!("wrapping");
    (unsafe { std::mem::transmute::<_, fn() -> i64>((FUNC_PTRS.lock().unwrap())[&sym]) })()
}
