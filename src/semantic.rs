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
    pub type_matches: Vec<Vec<Id>>,
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
            type_matches: Vec::new(),
        }
    }

    pub fn assign_top_level_types(&mut self) -> Result<(), CompileError> {
        // todo(chad): clone *sigh*
        for tl in self.parser.top_level.clone().iter() {
            self.assign_type(*tl, Coercion::None)?
        }

        self.unify_types();
        if !self.type_matches.is_empty() && !self.type_matches[0].is_empty() {
            CompileError::from_string(
                "Failed to unify types",
                self.parser.ranges[self.type_matches[0][0]],
            );
        }

        // for (index, ty) in self.types.iter().enumerate() {
        //     if ty == &Type::Unassigned {
        //         println!("Unassigned type for {}", self.parser.debug(index));
        //     }
        // }

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
                self.match_types(id, resolved);
                self.parser.node_is_addressable[id] = self.parser.node_is_addressable[resolved];
                Ok(())
            }
            Node::IntLiteral(_) => {
                self.types[id] = Type::Basic(BasicType::IntLiteral);

                let bt = match coercion {
                    Coercion::None => Some(BasicType::I64),
                    Coercion::Id(cid) => {
                        self.match_types(id, cid);
                        None
                    }
                    Coercion::Basic(bt) => Some(bt),
                };

                // todo(chad): check integer for overflow
                if let Some(bt) = bt {
                    match bt {
                        BasicType::Bool | BasicType::None => {
                            return Err(CompileError::from_string(
                                format!(
                                    "Cannot convert int literal into type {:?}",
                                    &self.types[id]
                                ),
                                self.parser.ranges[id],
                            ));
                        }
                        _ => self.types[id] = Type::Basic(bt),
                    };
                }

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
                if let Some(expr) = expr {
                    self.assign_type(*expr, Coercion::None)?;
                    self.match_types(id, *expr);
                }

                if let Some(ty) = ty {
                    self.assign_type(*ty, Coercion::None)?;
                    self.match_types(id, *ty);
                }

                Ok(())
            }
            Node::Set { name, expr, .. } => {
                let saved_lhs_assign = self.lhs_assign;
                self.lhs_assign = true;
                self.assign_type(*name, Coercion::None)?;
                self.lhs_assign = saved_lhs_assign;

                self.assign_type(*expr, Coercion::None)?;

                self.match_types(id, *expr);
                self.match_types(*name, *expr);

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

                        self.match_types(id, *ty);
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
                    Some(ty_id) => self.match_types(id, ty_id),
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

                self.match_types(*arg1, *arg2);
                self.match_types(id, *arg1);

                Ok(())
            }
            Node::LessThan(arg1, arg2)
            | Node::GreaterThan(arg1, arg2)
            | Node::EqualTo(arg1, arg2) => {
                self.assign_type(*arg1, coercion)?;
                self.assign_type(*arg2, coercion.or_id(*arg1))?;

                self.match_types(*arg1, *arg2);
                self.match_types(id, *arg1);

                Ok(())
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
                // if !ct_params.is_empty() && !is_specialized {
                //     return Ok(());
                // }

                for &ct_param in ct_params.iter() {
                    self.assign_type(ct_param, Coercion::None)?;
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
                self.match_types(id, *ty);
                Ok(())
            }
            Node::ValueParam { name: _, value, .. } => {
                self.assign_type(*value, coercion)?;
                self.match_types(id, *value);
                Ok(())
            }
            Node::Return(ret_id) => {
                let return_ty = *self.function_return_tys.borrow().last().unwrap();
                self.assign_type(*ret_id, Coercion::Id(return_ty))?;
                self.match_types(id, *ret_id);
                Ok(())
            }
            Node::If(cond_id, true_id, false_id) => {
                self.assign_type(*cond_id, Coercion::Basic(BasicType::Bool))?;
                self.assign_type(*true_id, Coercion::None)?;

                if false_id.is_some() {
                    self.assign_type(false_id.unwrap(), Coercion::Id(*true_id))?;
                }

                // todo(chad): enforce unit type for single-armed if stmts
                self.match_types(id, *true_id);

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
                        self.match_types(id, *pid);
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

                Ok(())
            }
            Node::Call {
                name,
                ct_params,
                params,
                is_indirect: _,
            } => {
                let resolved_func = self.resolve(*name)?;

                let is_ct = match &self.parser.nodes[resolved_func] {
                    Node::Func { ct_params, .. } => ct_params.len() > 0,
                    _ => match &self.types[resolved_func] {
                        // if it's a pure function type (already specialized) then it's not ct
                        Type::Func { .. } => false,
                        // otherwise what the heck are we trying to call?
                        _ => {
                            return Err(CompileError::from_string(
                                &format!("Cannot call a non-func: {:?}", &self.parser.nodes[id]),
                                self.parser.ranges[id],
                            ))
                        }
                    },
                };

                if is_ct {
                    // match given ct_params against declared ct_params
                    let given = ct_params;
                    let decl = match &self.parser.nodes[resolved_func] {
                        Node::Func { ct_params, .. } => ct_params.clone(),
                        _ => unreachable!(),
                    };

                    for (g, d) in given.iter().zip(decl.iter()) {
                        self.assign_type(*d, Coercion::None)?;
                        self.assign_type(*g, Coercion::Id(*d))?;
                        self.match_types(*g, *d);
                    }
                }

                match self.parser.nodes.get_mut(resolved_func).unwrap() {
                    Node::Func { is_specialized, .. } => {
                        *is_specialized = true;
                    }
                    _ => (),
                }

                let given = params;
                let decl = match &self.parser.nodes[resolved_func] {
                    Node::Func { params, .. } => params.clone(),
                    _ => match &self.types[resolved_func] {
                        Type::Func { input_tys, .. } => input_tys.clone(),
                        _ => unreachable!(),
                    },
                };

                for (g, d) in given.iter().zip(decl.iter()) {
                    self.assign_type(*d, Coercion::None)?;
                    self.assign_type(*g, Coercion::None)?;
                    self.match_types(*d, *g);
                }

                self.assign_type(*name, Coercion::None)?;

                match self.types[*name].clone() {
                    Type::Func { return_ty, .. } => self.match_types(id, return_ty),
                    _ => (),
                }

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

    fn find_type_array_index(&mut self, id: Id) -> usize {
        for (index, m) in self.type_matches.iter().enumerate() {
            if m.contains(&id) {
                return index;
            }
        }

        self.type_matches.push(vec![id]);
        self.type_matches.len() - 1
    }

    fn type_specificity(&self, id: Id) -> i16 {
        match &self.types[id] {
            Type::Unassigned | Type::Type => 0,
            Type::Basic(BasicType::IntLiteral) => 1,
            _ => 2,
        }
    }

    fn match_types(&mut self, ty1: Id, ty2: Id) {
        if self.types[ty2].is_concrete() && !self.types[ty1].is_concrete() {
            self.types[ty1] = self.types[ty2].clone();
        }
        if self.types[ty1].is_concrete() && !self.types[ty2].is_concrete() {
            self.types[ty2] = self.types[ty1].clone();
        }

        match (self.types[ty1].clone(), self.types[ty2].clone()) {
            (
                Type::Func {
                    return_ty: return_ty1,
                    input_tys: input_tys1,
                },
                Type::Func {
                    return_ty: return_ty2,
                    input_tys: input_tys2,
                },
            ) => {
                self.match_types(return_ty1, return_ty2);
                for (it1, it2) in input_tys1.iter().zip(input_tys2.iter()) {
                    self.match_types(*it1, *it2);
                }
            }
            (Type::Struct(st1), Type::Struct(st2)) => {
                for (ty1, ty2) in st1.iter().zip(st2.iter()) {
                    self.match_types(*ty1, *ty2);
                }
            }
            (_, _) => (),
        }

        // if ty1 == 24 && ty2 == 36 {
        // println!(
        //     "matching {} ({}) : {:?} with {} ({}) {:?}",
        //     ty1,
        //     self.parser.debug(ty1),
        //     &self.types[ty1],
        //     ty2,
        //     self.parser.debug(ty2),
        //     &self.types[ty2],
        // );
        // }

        let id1 = self.find_type_array_index(ty1);
        let id2 = self.find_type_array_index(ty2);

        if id1 != id2 {
            let lower = id1.min(id2);
            let upper = id1.max(id2);

            // todo(chad): partition or something
            let upper_matches = self.type_matches[upper].clone();
            self.type_matches[lower].extend(upper_matches);
            self.type_matches.remove(upper);
        }
    }

    fn unify_types(&mut self) {
        let mut to_clear = Vec::new();

        for uid in 0..self.type_matches.len() {
            // println!("\n********* Unifying *********");
            let tys = self.type_matches[uid]
                .iter()
                .map(|&ty| {
                    // println!(
                    //     "{:?} ({}) : {:?}",
                    //     &self.parser.nodes[ty],
                    //     self.parser.debug(ty),
                    //     &self.types[ty],
                    // );
                    (ty, self.type_specificity(ty))
                })
                .filter(|(_, spec)| *spec > 0)
                .filter(|(ty, _)| self.types[*ty as usize].is_concrete())
                // .max_by(|(_, spec1), (_, spec2)| spec1.cmp(spec2))
                .map(|(ty, _)| ty)
                .collect::<Vec<_>>();

            // for &ty in tys.iter() {
            //     if &self.types[ty] != &self.types[tys[0]] {
            //         todo!(
            //             "error message for types not equal: {:?} vs {:?}",
            //             &self.types[ty],
            //             &self.types[tys[0]],
            //         )
            //     }
            // }

            // set all types to whatever we unified to
            if let Some(&ty) = tys.first() {
                match self.types[ty as usize].clone() {
                    Type::Basic(BasicType::IntLiteral) => {
                        self.types[ty as usize] = Type::Basic(BasicType::I64);
                    }
                    _ => (),
                }
                // println!(
                //     "setting all to {:?} (from {:?})",
                //     self.types[ty as usize],
                //     self.parser.debug(ty as _)
                // );

                for id in self.type_matches[uid].iter() {
                    self.types[*id] = self.types[ty as usize].clone();
                }

                to_clear.push(uid);
            }
        }

        to_clear.reverse();
        for uid in to_clear {
            self.type_matches.remove(uid);
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
