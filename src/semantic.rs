use crate::parser::{BasicType, CompileError, Id, Node, Parser, Type};

type Sym = usize;

pub struct Semantic<'a> {
    pub parser: Parser<'a>,
    pub types: Vec<Type>,
    pub topo: Vec<Id>,
}

impl<'a> Semantic<'a> {
    pub fn new(parser: Parser<'a>) -> Self {
        let types = vec![Type::Unassigned; parser.nodes.len()];
        Self {
            parser,
            types,
            topo: Vec::new(),
        }
    }

    pub fn assign_top_level_types(&mut self) -> Result<(), CompileError> {
        // todo(chad): clone *sigh*
        for tl in self.parser.top_level.clone().iter() {
            self.assign_type(*tl)?
        }

        Ok(())
    }

    fn assign_type(&mut self, id: usize) -> Result<(), CompileError> {
        // idempotency
        match &self.types[id] {
            Type::Unassigned => {}
            _ => return Ok(()),
        }

        let node = &self.parser.nodes[id].clone();

        match node {
            Node::Symbol(sym) => {
                let resolved = self.scope_get(*sym, id)?;
                self.assign_type(resolved)?;
                self.types[id] = self.types[resolved].clone();
                Ok(())
            }
            Node::IntLiteral(_) => {
                self.types[id] = Type::Basic(BasicType::I64);
                Ok(())
            },
            Node::FloatLiteral(_) => {
                self.types[id] =  Type::Basic(BasicType::F64);
                Ok(())
            },
            Node::BoolLiteral(_) => {
                self.types[id] =  Type::Basic(BasicType::Bool);
                Ok(())
            }
            Node::StringLiteral(_) => {
                self.types[id] =  Type::String;
                Ok(())
            }
            Node::Let { name, ty, expr } => {
                self.parser.scope_insert(*name, id);

                self.assign_type(*ty)?;
                self.assign_type(*expr)?;

                if !self.types_match_id(*ty, *expr) {
                    let ty1 = self.types[*ty].clone();
                    let ty2 = self.types[*expr].clone();

                    Err(CompileError::from_string(
                        format!("Type mismatch: {:?} vs {:?}", ty1, ty2),
                        self.parser.ranges[id],
                    ))
                } else {
                    self.types[id] =  self.types[*expr].clone();
                    Ok(())
                }
            }
            Node::Add(arg1, arg2) | Node::Sub(arg1, arg2) | Node::Mul(arg1, arg2) | Node::Div(arg1, arg2) => {
                self.assign_type(*arg1)?;
                self.assign_type(*arg2)?;

                let ty1: Type = self.types[*arg1].clone();
                let ty2: Type = self.types[*arg2].clone();

                if !self.types_match_ty(&ty1, &ty2) {
                    Err(CompileError::from_string(
                        format!("Type mismatch: {:?} vs {:?}", ty1, ty2),
                        self.parser.ranges[id],
                    ))
                } else {
                    self.types[id] =  ty1;
                    Ok(())
                }
            }
            Node::LessThan(arg1, arg2) | Node::GreaterThan(arg1, arg2) | Node::EqualTo(arg1, arg2) => {
                self.assign_type(*arg1)?;
                self.assign_type(*arg2)?;

                let ty1: Type = self.types[*arg1].clone();
                let ty2: Type = self.types[*arg2].clone();

                if !self.types_match_ty(&ty1, &ty2) {
                    Err(CompileError::from_string(
                        format!("Type mismatch: {:?} vs {:?}", ty1, ty2),
                        self.parser.ranges[id],
                    ))
                } else {
                    self.types[id] =  Type::Basic(BasicType::Bool);
                    Ok(())
                }
            }
            Node::And(arg1, arg2) | Node::Or(arg1, arg2) => {
                self.assign_type(*arg1)?;
                self.assign_type(*arg2)?;

                Ok(())
            }
            Node::Not(arg1) => {
                self.assign_type(*arg1)?;
                Ok(())
            }
            Node::Func {
                name: _,   //: Sym,
                scope: _,  //: Id,
                params,    // : Vec<Id>,
                return_ty, //: Id,
                stmts,     //: Vec<Id>,
            } => {
                self.assign_type(*return_ty)?;
                for &param in params.iter() {
                    self.assign_type(param)?;
                }

                for &stmt in stmts.iter() {
                    self.assign_type(stmt)?;
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
                self.assign_type(*ty)?;
                self.types[id] =  self.types[*ty].clone();
                Ok(())
            }
            Node::Return(ret_id) => {
                self.assign_type(*ret_id)?;
                self.types[id] =  self.types[*ret_id].clone();
                Ok(())
            }
            Node::If(cond_id, true_id, false_id) => {
                self.assign_type(*cond_id)?;
                self.assign_type(*true_id)?;

                // todo(chad): assert boolean type for cond_id

                if false_id.is_some() {
                    self.assign_type(false_id.unwrap())?;

                    if !self.types_match_id(*true_id, false_id.unwrap()) {
                        let ty1 = self.types[*true_id].clone();
                        let ty2 = self.types[false_id.unwrap()].clone();

                        return Err(CompileError::from_string(
                            format!("Types for 'if' don't match: {:?} vs {:?}", ty1, ty2), 
                            self.parser.ranges[id]
                        ));
                    }
                }

                // todo(chad): enforce unit type for single-armed if stmts
                self.types[id] = self.types[*true_id].clone();

                Ok(())
            }
            Node::While(cond, stmts) => {
                self.assign_type(*cond)?;
                for &stmt in stmts {
                    self.assign_type(stmt)?;
                }

                Ok(())
            }
            Node::Store {
                ptr,  // : Id,
                expr, // : Id,
            } => {
                // type of name
                // type of expr, using type of name as coercion
                self.assign_type(*ptr)?;

                let resolved_ty = match &self.types[*ptr] {
                    Type::Pointer(p) => Ok(self.types[*p].clone()),
                    _ => Err(CompileError::from_string(format!("expected to store into a pointer, not a {:?}", &self.types[*ptr]), self.parser.ranges[id]))
                }?;

                self.assign_type(*expr)?;
                self.types[id] =  resolved_ty;

                Ok(())
            }
            Node::PtrCast { ty, expr } => {
                self.assign_type(*ty)?;
                self.assign_type(*expr)?;
                match &self.types[*expr] {
                    Type::Pointer(_) => {
                        self.types[id] =  Type::Pointer(*ty);
                        Ok(())
                    }
                    _ => Err(CompileError::from_string("Cannot ptr-cast a non-ptr", self.parser.ranges[*expr]))
                }
            }
            Node::TypeLiteral(ty) => {
                match ty {
                    Type::Basic(_) => {
                        self.types[id] =  ty.clone().into();
                    }
                    Type::Pointer(ty) => {
                        self.assign_type(*ty)?;
                        self.types[id] =  Type::Pointer(*ty);
                    }
                    Type::String => {
                        self.types[id] =  Type::String;
                    }
                    Type::Func {
                        return_ty,
                        input_tys,
                    } => {
                        self.assign_type(*return_ty)?;
                        for &input_ty in input_tys {
                            self.assign_type(input_ty)?;
                        }

                        self.types[id] =  ty.clone().into();
                    }
                    Type::Unassigned => unreachable!()
                }

                Ok(())
            }
            Node::Call { name, args, is_macro, is_indirect } => if *is_macro {
                for arg in args {
                    self.assign_type(*arg)?;
                }

                self.assign_type(*name)?;
                self.types[id] =  Type::Basic(BasicType::None);
                Ok(())
            } else if *is_indirect {
                for arg in args {
                    self.assign_type(*arg)?;
                }

                self.assign_type(*name)?;

                match &self.types[*name].clone() {
                    Type::Func {
                        return_ty,
                        input_tys: _,
                    } => {
                        // TODO: check arg types against declared arg types

                        self.types[id] =  self.types[*return_ty].clone();
                        Ok(())
                    }
                    _ => Err(CompileError::from_string(
                        "Failed to get type of function in call",
                        self.parser.ranges[id],
                    )),
                }
            } else {
                for arg in args {
                    self.assign_type(*arg)?;
                }

                self.assign_type(*name)?;

                match &self.types[*name].clone() {
                    Type::Func {
                        return_ty,
                        input_tys: _,
                    } => {
                        // TODO: check arg types against declared arg types

                        self.types[id] =  self.types[*return_ty].clone();
                        Ok(())
                    }
                    _ => Err(CompileError::from_string(
                        "Failed to get type of function in call",
                        self.parser.ranges[id],
                    )),
                }
            }
            Node::Extern {
                name: _,   // Sym,
                params,    // Vec<Id>,
                return_ty, // Id,
            } => {
                for &input in params.iter() {
                    self.assign_type(input)?;
                }
                self.assign_type(*return_ty)?;

                self.types[id] = Type::Func {
                    return_ty: *return_ty,
                    input_tys: params.clone(),
                };

                Ok(())
            }
            // _ => Err(CompileError::from_string(
            //     format!(
            //         "Cannot coerce type for AST node {:?}",
            //         &self.parser.nodes[id]
            //     ),
            //     self.parser.ranges[id],
            // )),
        }
    }

    fn types_match_id(&self, ty1: Id, ty2: Id) -> bool {
        let ty1 = self.types[ty1].clone();
        let ty2 = self.types[ty2].clone();
        self.types_match_ty(&ty1, &ty2)
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
