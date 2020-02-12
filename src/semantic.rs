use crate::parser::{BasicType, CompileError, Id, Node, Parser, Type};

type Sym = usize;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
enum Coercion {
    None,
    Id(Id),
    Basic(BasicType),
}

impl Coercion {
    fn or_id(&self, id: Id) -> Coercion {
        match self {
            Coercion::None => Coercion::Id(id),
            _ => *self,
        }
    }
}

pub struct Semantic<'a> {
    pub parser: Parser<'a>,
    pub types: Vec<Type>,
    pub topo: Vec<Id>,
    // todo(chad): probably not a big deal, but try to find a workaround
    pub function_return_tys: Rc<RefCell<Vec<Id>>>,
    pub lhs_assign: bool,
}

impl<'a> Semantic<'a> {
    pub fn new(parser: Parser<'a>) -> Self {
        let types = vec![Type::Unassigned; parser.nodes.len()];
        Self {
            parser,
            types,
            topo: Vec::new(),
            function_return_tys: Rc::new(RefCell::new(Vec::new())),
            lhs_assign: false,
        }
    }

    pub fn assign_top_level_types(&mut self) -> Result<(), CompileError> {
        // todo(chad): clone *sigh*
        for tl in self.parser.top_level.clone().iter() {
            self.assign_type(*tl, Coercion::None)?
        }

        Ok(())
    }

    fn assign_type(&mut self, id: usize, coercion: Coercion) -> Result<(), CompileError> {
        // idempotency
        match &self.types[id] {
            Type::Unassigned => {}
            _ => return Ok(()),
        }

        let node = &self.parser.nodes[id].clone();

        match node {
            Node::Symbol(sym) => {
                let resolved = self.scope_get(*sym, id)?;
                self.assign_type(resolved, coercion)?;
                self.types[id] = self.types[resolved].clone();
                self.parser.node_is_addressable[id] = self.parser.node_is_addressable[resolved];
                Ok(())
            }
            Node::IntLiteral(_) => {
                let bt = match coercion {
                    Coercion::None => BasicType::I64,
                    Coercion::Id(cid) => match self.types[cid].clone() {
                        Type::Basic(bt) => bt,
                        Type::Type => {
                            self.types[cid] = Type::Basic(BasicType::I64);
                            BasicType::I64
                        }
                        _ => {
                            return Err(CompileError::from_string(
                                format!(
                                    "Cannot convert int literal into type {:?}",
                                    &self.types[id]
                                ),
                                self.parser.ranges[id],
                            ));
                        }
                    },
                    Coercion::Basic(bt) => bt,
                };

                // todo(chad): check integer for overflow
                match bt {
                    BasicType::Bool | BasicType::None => {
                        return Err(CompileError::from_string(
                            format!("Cannot convert int literal into type {:?}", &self.types[id]),
                            self.parser.ranges[id],
                        ));
                    }
                    _ => self.types[id] = Type::Basic(bt),
                };

                Ok(())
            }
            Node::FloatLiteral(_) => {
                self.types[id] = Type::Basic(BasicType::F64);
                Ok(())
            }
            Node::BoolLiteral(_) => {
                self.types[id] = Type::Basic(BasicType::Bool);
                Ok(())
            }
            Node::StringLiteral(_) => {
                self.types[id] = Type::String;
                Ok(())
            }
            Node::Let { name: _, ty, expr } => {
                let coercion = match ty {
                    Some(ty) => {
                        self.assign_type(*ty, Coercion::None)?;
                        Coercion::Id(*ty)
                    }
                    None => Coercion::None,
                };

                if expr.is_some() {
                    self.assign_type(expr.unwrap(), coercion)?;
                }

                match (ty, expr) {
                    (Some(ty), _) => {
                        self.types[id] = self.types[*ty].clone();
                        Ok(())
                    }
                    (_, Some(expr)) => {
                        self.types[id] = self.types[*expr].clone();
                        Ok(())
                    }
                    (None, None) => Err(CompileError::from_string(
                        "'let' binding with unbound type and no expression",
                        self.parser.ranges[id],
                    )),
                }
            }
            Node::Set { name, expr, .. } => {
                let saved_lhs_assign = self.lhs_assign;
                self.lhs_assign = true;
                self.assign_type(*name, Coercion::None)?;
                self.lhs_assign = saved_lhs_assign;

                self.assign_type(*expr, Coercion::Id(*name))?;

                // if name is a load, then we are doing a store
                let is_load = match &self.parser.nodes[*name] {
                    Node::Load(_) => true,
                    _ => false,
                };

                if is_load {
                    match &mut self.parser.nodes[id] {
                        Node::Set { is_store, .. } => *is_store = true,
                        _ => {}
                    }
                }

                Ok(())
            }
            Node::Field {
                base,
                field_name,
                field_index: _,
                ..
            } => {
                self.assign_type(*base, Coercion::None)?;
                self.parser.node_is_addressable[*base] = true;

                // todo(chad): deref more than once?
                let unpointered_ty = match &self.types[*base] {
                    Type::Pointer(id) => *id,
                    _ => *base,
                };

                if self.lhs_assign {
                    match self.parser.nodes.get_mut(id).unwrap() {
                        Node::Field { is_assignment, .. } => {
                            *is_assignment = true;
                        }
                        _ => unreachable!(),
                    }
                }

                let params = self.types[unpointered_ty]
                    .as_struct_params()
                    .ok_or(CompileError::from_string(
                        format!(
                            "Expected struct or enum, got {:?}",
                            self.parser.debug(unpointered_ty)
                        ),
                        self.parser.ranges[*base],
                    ))?
                    .iter()
                    .enumerate()
                    .map(|(_index, id)| {
                        let (name, ty, index) = self.parser.nodes[*id].as_param().unwrap();
                        (name, (ty, index))
                    })
                    .collect::<BTreeMap<_, _>>();

                match params.get(field_name) {
                    Some((ty, index)) => {
                        // set the field index
                        match &mut self.parser.nodes[id] {
                            Node::Field { field_index, .. } => *field_index = *index,
                            _ => unreachable!(),
                        }

                        self.types[id] = self.types[*ty].clone();
                        Ok(())
                    }
                    _ => Err(CompileError::from_string(
                        "field not found",
                        self.parser.ranges[id],
                    )),
                }
            }
            Node::StructLiteral { name: _, params } => {
                // todo(chad): @Performance this does not always need to be true, see comment in backend (compile_id on Node::StructLiteral)
                self.parser.node_is_addressable[id] = true;

                // todo(chad): @Cleanup this is ugly
                let (params_map, lit_ty) = match coercion {
                    Coercion::Id(id) => match &self.types[id] {
                        Type::Struct(params) => (
                            Some(
                                params
                                    .iter()
                                    .map(|p| {
                                        let (name, id, index) =
                                            self.parser.nodes[*p].as_param().unwrap();
                                        (name, (id, index))
                                    })
                                    .collect::<BTreeMap<_, _>>(),
                            ),
                            Some(id),
                        ),
                        Type::Enum(params) => (
                            Some(
                                params
                                    .iter()
                                    .map(|p| {
                                        let (name, id, index) =
                                            self.parser.nodes[*p].as_param().unwrap();
                                        (name, (id, index))
                                    })
                                    .collect::<BTreeMap<_, _>>(),
                            ),
                            Some(id),
                        ),
                        _ => {
                            return Err(CompileError::from_string(
                                "Cannot match struct literal with this type",
                                self.parser.ranges[id],
                            ));
                        }
                    },
                    _ => (None, None),
                };

                for param in params {
                    let (name, _, _) = self.parser.nodes[*param].as_value_param().unwrap();

                    let coercion = match &params_map {
                        Some(params_map) => {
                            let (id, param_index) = params_map
                                .get(&name.expect("todo: handle missing name"))
                                .expect("todo: handle wrong param");

                            match self.parser.nodes.get_mut(*param).unwrap() {
                                Node::ValueParam { index, .. } => {
                                    *index = *param_index;
                                }
                                _ => unreachable!(),
                            }

                            Coercion::Id(*id)
                        }
                        None => Coercion::None,
                    };

                    self.assign_type(*param, coercion)?;
                }

                match lit_ty {
                    Some(ty_id) => self.types[id] = self.types[ty_id].clone(),
                    None => self.types[id] = Type::Struct(params.clone()),
                }

                Ok(())
            }
            Node::Add(arg1, arg2)
            | Node::Sub(arg1, arg2)
            | Node::Mul(arg1, arg2)
            | Node::Div(arg1, arg2) => {
                self.assign_type(*arg1, coercion)?;
                self.assign_type(*arg2, coercion.or_id(*arg1))?;

                let ty1: Type = self.types[*arg1].clone();
                let ty2: Type = self.types[*arg2].clone();

                if !self.types_match_ty(&ty1, &ty2) {
                    Err(CompileError::from_string(
                        format!("Type mismatch: {:?} vs {:?}", ty1, ty2),
                        self.parser.ranges[id],
                    ))
                } else {
                    self.types[id] = ty1;
                    Ok(())
                }
            }
            Node::LessThan(arg1, arg2)
            | Node::GreaterThan(arg1, arg2)
            | Node::EqualTo(arg1, arg2) => {
                self.assign_type(*arg1, coercion)?;
                self.assign_type(*arg2, coercion.or_id(*arg1))?;

                let ty1: Type = self.types[*arg1].clone();
                let ty2: Type = self.types[*arg2].clone();

                if !self.types_match_ty(&ty1, &ty2) {
                    Err(CompileError::from_string(
                        format!("Type mismatch: {:?} vs {:?}", ty1, ty2),
                        self.parser.ranges[id],
                    ))
                } else {
                    self.types[id] = Type::Basic(BasicType::Bool);
                    Ok(())
                }
            }
            Node::And(arg1, arg2) | Node::Or(arg1, arg2) => {
                self.assign_type(*arg1, coercion)?;
                self.assign_type(*arg2, coercion.or_id(*arg1))?;

                Ok(())
            }
            Node::Not(arg1) => {
                self.assign_type(*arg1, Coercion::Basic(BasicType::Bool))?;
                Ok(())
            }
            Node::Func {
                name: _,   //: Sym,
                scope: _,  //: Id,
                ct_params, //: Vec<Id>,
                params,    //: Vec<Id>,
                return_ty, //: Id,
                stmts,     //: Vec<Id>,
                is_specialized,
            } => {
                if !ct_params.is_empty() && !is_specialized {
                    return Ok(());
                }

                self.assign_type(*return_ty, Coercion::None)?;

                let function_return_tys = self.function_return_tys.clone();
                function_return_tys.borrow_mut().push(*return_ty);
                defer! {{
                    function_return_tys.borrow_mut().pop();
                }};

                for &param in params.iter() {
                    self.assign_type(param, Coercion::None)?;
                }

                for &stmt in stmts.iter() {
                    self.assign_type(stmt, Coercion::None)?;
                }

                self.types[id] = Type::Func {
                    return_ty: *return_ty,
                    input_tys: params.clone(),
                };

                self.topo.push(id);

                Ok(())
            }
            Node::DeclParam {
                name: _,  // Sym
                ty,       // Id
                index: _, //  u16
            } => {
                self.assign_type(*ty, Coercion::None)?;
                self.types[id] = self.types[*ty].clone();
                Ok(())
            }
            Node::ValueParam { name: _, value, .. } => {
                self.assign_type(*value, coercion)?;
                self.types[id] = self.types[*value].clone();
                Ok(())
            }
            Node::Return(ret_id) => {
                let return_ty = *self.function_return_tys.borrow().last().unwrap();
                self.assign_type(*ret_id, Coercion::Id(return_ty))?;
                self.types[id] = self.types[*ret_id].clone();
                Ok(())
            }
            Node::If(cond_id, true_id, false_id) => {
                self.assign_type(*cond_id, Coercion::Basic(BasicType::Bool))?;
                self.assign_type(*true_id, Coercion::None)?;

                if false_id.is_some() {
                    self.assign_type(false_id.unwrap(), Coercion::Id(*true_id))?;
                }

                // todo(chad): enforce unit type for single-armed if stmts
                self.types[id] = self.types[*true_id].clone();

                Ok(())
            }
            Node::While(cond, stmts) => {
                self.assign_type(*cond, Coercion::Basic(BasicType::Bool))?;
                for &stmt in stmts {
                    self.assign_type(stmt, Coercion::None)?;
                }

                Ok(())
            }
            Node::Ref(ref_id) => {
                // todo(chad): coercion
                self.assign_type(*ref_id, Coercion::None)?;
                self.types[id] = Type::Pointer(*ref_id);

                Ok(())
            }
            Node::Load(load_id) => {
                // todo(chad): coercion
                self.assign_type(*load_id, Coercion::None)?;

                match &self.types[*load_id] {
                    Type::Pointer(pid) => {
                        self.types[id] = self.types[*pid].clone();
                        Ok(())
                    }
                    _ => Err(CompileError::from_string(
                        "Cannot load a non-pointer",
                        self.parser.ranges[id],
                    )),
                }
            }
            Node::TypeLiteral(ty) => {
                match ty {
                    Type::Basic(_) => {
                        self.types[id] = ty.clone();
                    }
                    Type::Pointer(ty) => {
                        self.assign_type(*ty, Coercion::None)?;
                        self.types[id] = Type::Pointer(*ty);
                    }
                    Type::String => {
                        self.types[id] = Type::String;
                    }
                    Type::Type => {
                        self.types[id] = Type::Type;
                    }
                    Type::Func {
                        return_ty,
                        input_tys,
                    } => {
                        self.assign_type(*return_ty, Coercion::None)?;
                        for &input_ty in input_tys {
                            self.assign_type(input_ty, Coercion::None)?;
                        }

                        self.types[id] = ty.clone();
                    }
                    Type::Struct(params) => {
                        for ty in params {
                            self.assign_type(*ty, Coercion::None)?;
                        }

                        self.types[id] = ty.clone();
                    }
                    Type::Enum(params) => {
                        for ty in params {
                            self.assign_type(*ty, Coercion::None)?;
                        }

                        self.types[id] = ty.clone();
                    }
                    Type::Unassigned => unreachable!(),
                }

                // if we match against a Type, then set it to whatever we are
                // todo(chad): this is temporary, need to do the whole array matching thing in the future
                match coercion {
                    Coercion::Id(cid) => match self.types[cid].clone() {
                        Type::Type => {
                            println!("setting {} to {:?}", cid, &self.types[id]);
                            self.types[cid] = self.types[id].clone();
                        }
                        _ => (),
                    },
                    _ => (),
                }

                Ok(())
            }
            Node::Call {
                name,
                ct_params,
                params,
                is_indirect: _,
            } => {
                let resolved_func = self.resolve(*name)?;
                let func = self.parser.nodes[resolved_func].clone();

                let is_ct = match &func {
                    Node::Func { ct_params, .. } => ct_params.len() > 0,
                    _ => {
                        return Err(CompileError::from_string(
                            "Cannot call a non-func",
                            self.parser.ranges[id],
                        ))
                    }
                };

                if is_ct {
                    // match given ct_params against declared ct_params
                    let given = ct_params;
                    let decl = match &func {
                        Node::Func { ct_params, .. } => ct_params.clone(),
                        _ => unreachable!(),
                    };

                    for (g, d) in given.iter().zip(decl.iter()) {
                        self.assign_type(*d, Coercion::None)?;
                        self.assign_type(*g, Coercion::Id(*d))?;
                    }
                }

                match self.parser.nodes.get_mut(resolved_func).unwrap() {
                    Node::Func { is_specialized, .. } => {
                        *is_specialized = true;
                    }
                    _ => unreachable!(),
                }

                let given = params;
                let decl = match func {
                    Node::Func { params, .. } => params.clone(),
                    _ => unreachable!(),
                };

                for (g, d) in given.iter().zip(decl.iter()) {
                    // self.assign_type(*d, Coercion::None)?;
                    self.assign_type(*g, Coercion::Id(*d))?;
                }

                self.assign_type(*name, Coercion::None)?;

                Ok(())
            }
            Node::Struct { name: _, params } => {
                for param in params {
                    self.assign_type(*param, Coercion::None)?;
                }
                self.types[id] = Type::Struct(params.clone());

                Ok(())
            }
            Node::Enum { name: _, params } => {
                for param in params {
                    self.assign_type(*param, Coercion::None)?;
                }
                self.types[id] = Type::Enum(params.clone());

                Ok(())
            }
            _ => Err(CompileError::from_string(
                format!(
                    "Cannot coerce type for AST node {:?}",
                    &self.parser.nodes[id]
                ),
                self.parser.ranges[id],
            )),
        }
    }

    fn resolve(&self, id: Id) -> Result<Id, CompileError> {
        match &self.parser.nodes[id] {
            Node::Symbol(sym) => self.scope_get(*sym, id),
            _ => Ok(id),
        }
    }

    fn types_match_ty(&self, ty1: &Type, ty2: &Type) -> bool {
        match (ty1, ty2) {
            (Type::Pointer(pt1), Type::Pointer(pt2)) => self.types[*pt1] == self.types[*pt2],
            _ => ty1 == ty2,
        }
    }

    pub fn scope_get(&self, sym: Sym, id: Id) -> Result<Id, CompileError> {
        self.parser
            .scope_get(sym, id)
            .ok_or(CompileError::from_string(
                format!(
                    "Undeclared identifier {}",
                    self.parser.resolve_sym_unchecked(sym)
                ),
                self.parser.ranges[id],
            ))
    }
}
