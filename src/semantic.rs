use crate::parser::{BasicType, CompileError, Id, Node, Parser, Type};

type Sym = usize;

use std::cell::RefCell;
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
    // todo(chad): probably not a big deal, but maybe try to find a workaround
    pub function_return_tys: Rc<RefCell<Vec<Id>>>,
}

impl<'a> Semantic<'a> {
    pub fn new(parser: Parser<'a>) -> Self {
        let types = vec![Type::Unassigned; parser.nodes.len()];
        Self {
            parser,
            types,
            topo: Vec::new(),
            function_return_tys: Rc::new(RefCell::new(Vec::new())),
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
                Ok(())
            }
            Node::IntLiteral(_) => {
                let bt = match coercion {
                    Coercion::None => BasicType::I64,
                    Coercion::Id(id) => match &self.types[id] {
                        Type::Basic(bt) => *bt,
                        _ => {
                            return Err(CompileError::from_string(
                                "Cannot convert int literal into type",
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
                            "Cannot convert int literal into type",
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
            Node::Let { name, ty, expr } => {
                self.parser.scope_insert(*name, id);

                self.assign_type(*ty, Coercion::None)?;
                self.assign_type(*expr, Coercion::Id(*ty))?;

                self.types[id] = self.types[*expr].clone();
                Ok(())
            }
            Node::Set { name, expr } => {
                self.assign_type(*name, Coercion::None)?;
                self.assign_type(*expr, Coercion::Id(*name))?;

                self.types[id] = self.types[*expr].clone();
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
                params,    // : Vec<Id>,
                return_ty, //: Id,
                stmts,     //: Vec<Id>,
            } => {
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
            Node::FuncParam {
                name: _,  // Sym
                ty,       // Id
                index: _, //  u16
            } => {
                self.assign_type(*ty, Coercion::None)?;
                self.types[id] = self.types[*ty].clone();
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
            // Node::Store {
            //     ptr,  // : Id,
            //     expr, // : Id,
            // } => {
            //     // type of name
            //     // type of expr, using type of name as coercion
            //     self.assign_type(*ptr)?;

            //     let resolved_ty = match &self.types[*ptr] {
            //         Type::Pointer(p) => Ok(self.types[*p].clone()),
            //         _ => Err(CompileError::from_string(format!("expected to store into a pointer, not a {:?}", &self.types[*ptr]), self.parser.ranges[id]))
            //     }?;

            //     self.assign_type(*expr)?;
            //     self.types[id] =  resolved_ty;

            //     Ok(())
            // }
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
            // Node::PtrCast { ty, expr } => {
            //     self.assign_type(*ty)?;
            //     self.assign_type(*expr)?;
            //     match &self.types[*expr] {
            //         Type::Pointer(_) => {
            //             self.types[id] =  Type::Pointer(*ty);
            //             Ok(())
            //         }
            //         _ => Err(CompileError::from_string("Cannot ptr-cast a non-ptr", self.parser.ranges[*expr]))
            //     }
            // }
            Node::TypeLiteral(ty) => {
                match ty {
                    Type::Basic(_) => {
                        self.types[id] = ty.clone().into();
                    }
                    Type::Pointer(ty) => {
                        self.assign_type(*ty, Coercion::None)?;
                        self.types[id] = Type::Pointer(*ty);
                    }
                    Type::String => {
                        self.types[id] = Type::String;
                    }
                    Type::Func {
                        return_ty,
                        input_tys,
                    } => {
                        self.assign_type(*return_ty, Coercion::None)?;
                        for &input_ty in input_tys {
                            self.assign_type(input_ty, Coercion::None)?;
                        }

                        self.types[id] = ty.clone().into();
                    }
                    Type::Unassigned => unreachable!(),
                }

                Ok(())
            }
            Node::Call {
                name,
                params,
                is_macro,
                is_indirect: _,
            } => {
                if *is_macro {
                    self.assign_type(*name, Coercion::None)?;

                    // todo(chad): get function type from name, match param types based on it
                    for param in params {
                        self.assign_type(*param, Coercion::None)?;
                    }

                    self.types[id] = Type::Basic(BasicType::None);
                    Ok(())
                } else {
                    self.assign_type(*name, Coercion::None)?;

                    let param_tys;
                    match &self.types[*name].clone() {
                        Type::Func {
                            return_ty,
                            input_tys,
                        } => {
                            self.types[id] = self.types[*return_ty].clone();
                            param_tys = input_tys.clone();
                        }
                        _ => {
                            return Err(CompileError::from_string(
                                "Failed to get type of function in call",
                                self.parser.ranges[id],
                            ));
                        }
                    }

                    for (param, param_ty) in params.iter().zip(param_tys.iter()) {
                        self.assign_type(*param, Coercion::Id(*param_ty))?;
                    }

                    Ok(())
                }
            }
            // Node::Extern {
            //     name: _,   // Sym,
            //     params,    // Vec<Id>,
            //     return_ty, // Id,
            // } => {
            //     for &input in params.iter() {
            //         self.assign_type(input, Coercion::None)?;
            //     }
            //     self.assign_type(*return_ty, Coercion::None)?;

            //     self.types[id] = Type::Func {
            //         return_ty: *return_ty,
            //         input_tys: params.clone(),
            //     };

            //     Ok(())
            // }
            _ => Err(CompileError::from_string(
                format!(
                    "Cannot coerce type for AST node {:?}",
                    &self.parser.nodes[id]
                ),
                self.parser.ranges[id],
            )),
        }
    }

    // fn types_match_id(&self, ty1: Id, ty2: Id) -> bool {
    //     let ty1 = self.types[ty1].clone();
    //     let ty2 = self.types[ty2].clone();
    //     self.types_match_ty(&ty1, &ty2)
    // }

    fn types_match_ty(&self, ty1: &Type, ty2: &Type) -> bool {
        match (ty1, ty2) {
            (Type::Pointer(pt1), Type::Pointer(pt2)) => self.types[*pt1] == self.types[*pt2],
            _ => ty1 == ty2,
        }
    }

    // fn coerce(&self, ty: Type, coercion: Coercion, range: &Range) -> Result<Type, CompileError> {
    //     let coercion = match coercion {
    //         Coercion::None => {
    //             return Ok(ty);
    //         }
    //         Coercion::Id(id) => self.types[id].clone(),
    //         Coercion::Basic(bt) => Type::Basic(bt),
    //     };

    //     let err = Err(CompileError::from_string(
    //         format!("Incompatible types: {:?} and {:?}", ty, coercion),
    //         *range,
    //     ));

    //     match (ty, coercion) {
    //         (Type::Basic(btt), Type::Basic(btc)) => match (btt, btc) {
    //             (BasicType::None, btc) => Ok(Type::Basic(btc)),
    //             (btt, BasicType::None) => Ok(Type::Basic(btt)),
    //             (BasicType::Bool, BasicType::Bool) => Ok(Type::Basic(BasicType::Bool)),
    //             (BasicType::I8, BasicType::I8) => Ok(Type::Basic(BasicType::I8)),
    //             (BasicType::I8, BasicType::I16) => Ok(Type::Basic(BasicType::I16)),
    //             (BasicType::I8, BasicType::I32) => Ok(Type::Basic(BasicType::I32)),
    //             (BasicType::I8, BasicType::I64) => Ok(Type::Basic(BasicType::I64)),
    //             (BasicType::I16, BasicType::I16) => Ok(Type::Basic(BasicType::I16)),
    //             (BasicType::I16, BasicType::I32) => Ok(Type::Basic(BasicType::I32)),
    //             (BasicType::I16, BasicType::I64) => Ok(Type::Basic(BasicType::I64)),
    //             (BasicType::I32, BasicType::I32) => Ok(Type::Basic(BasicType::I32)),
    //             (BasicType::I32, BasicType::I64) => Ok(Type::Basic(BasicType::I64)),
    //             (BasicType::I64, BasicType::I64) => Ok(Type::Basic(BasicType::I64)),
    //             _ => err,
    //         },
    //         (Type::String, Type::String) => Ok(Type::String),
    //         (Type::Unassigned, coercion) => Ok(coercion),
    //         (ty, Type::Unassigned) => Ok(ty),
    //         (Type::Pointer(_), Type::Pointer(_)) => todo!("handle pointer type coercion"),
    //         (Type::Func { .. }, Type::Func { .. }) => todo!("handle func type coercion"),
    //         _ => err,
    //     }
    // }

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
