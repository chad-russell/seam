use crate::parser::{BasicType, CompileError, Id, Node, Parser, Type, IdVec};

type Sym = usize;

use std::collections::BTreeMap;

pub struct Semantic<'a> {
    pub parser: Parser<'a>,
    pub types: Vec<Type>,
    pub topo: Vec<Id>,
    pub lhs_assign: bool,
    pub type_matches: Vec<Vec<Id>>,
    pub macro_phase: bool,
    pub macro_expansion_site: Option<Id>, // todo(chad): make this just Id once we're confident it's always set
    pub unquote_values: Vec<Node>,        // todo(chad): @ConstValue
}

impl<'a> Semantic<'a> {
    pub fn new(parser: Parser<'a>) -> Self {
        let types = vec![Type::Unassigned; parser.nodes.len()];
        Self {
            parser,
            types,
            topo: Vec::new(),
            lhs_assign: false,
            type_matches: Vec::new(),
            macro_phase: false,
            macro_expansion_site: None,
            unquote_values: Vec::new(),
        }
    }

    // fn handle_macros(&mut self) -> Result<(), CompileError> {
    //     self.macro_phase = true;

    //     let mut backend = Backend::new(self);
    //     for mac in backend.semantic.parser.macros.clone() {
    //         backend.semantic.assign_type(mac)?;
    //         backend.compile()?;
    //     }

    //     // todo(chad): ugh clone... this one could actually be expensive...
    //     for call in backend.semantic.parser.calls.clone() {
    //         let (name, params) = match backend.semantic.parser.nodes[call] {
    //             // todo(chad): disallow generic macros?
    //             Node::Call { name, params, .. } => (name, params),
    //             _ => unreachable!(),
    //         };

    //         let resolved_func = backend.semantic.resolve(name)?;
    //         let is_macro = match &backend.semantic.parser.nodes[resolved_func] {
    //             Node::Func { is_macro, .. } => *is_macro,
    //             _ => false,
    //         };

    //         if !is_macro {
    //             continue;
    //         }

    //         let (name, decl) = match &backend.semantic.parser.nodes[resolved_func] {
    //             Node::Func { name, params, .. } => (*name, *params),
    //             _ => unreachable!(),
    //         };

    //         // resolve any constant params
    //         let given = backend.semantic.parser.id_vec(params).clone();
    //         let decl = backend.semantic.parser.id_vec(decl).clone();

    //         for (g, d) in given.iter().zip(decl.iter()) {
    //             backend.semantic.assign_type(*d)?;
    //             backend.semantic.assign_type(*g)?;
    //             backend.semantic.match_types(*g, *d);
    //         }

    //         backend.semantic.unify_types(true)?;

    //         let maybe_hoisted: *const u8 = if !given.is_empty() {
    //             // todo(chad): build constant 'includes', i.e. what else has to be pulled into the hoisted fn to support the const param
    //             backend.build_hoisted_function(given, name)?
    //         } else {
    //             (crate::backend::FUNC_PTRS.lock().unwrap())[&name].0
    //         };

    //         let f: fn(*const Semantic) = unsafe { std::mem::transmute(maybe_hoisted) };

    //         backend.semantic.macro_expansion_site = Some(call);

    //         backend.semantic.unquote_values.clear();
    //         backend.semantic.parser.parsed_unquotes.clear();
    //         f(backend.semantic as _);

    //         let unquotes = backend.semantic.parser.parsed_unquotes.clone();
    //         let values = backend.semantic.unquote_values.clone();
    //         for (unquote, value) in unquotes.iter().zip(values.iter()) {
    //             backend.semantic.parser.nodes[*unquote] = *value;
    //         }
    //     }

    //     self.macro_phase = false;
    //     self.topo.clear();

    //     Ok(())
    // }

    pub fn assign_type(&mut self, id: usize) -> Result<(), CompileError> {
        // idempotency
        let type_is_assigned = match &self.types[id] {
            Type::Unassigned => false,
            _ => true,
        };

        let is_poly = match &self.parser.nodes[id] {
            Node::Func { ct_params, .. } => ct_params.is_some(),
            Node::Struct { ct_params, .. } => ct_params.is_some(),
            // Node::Enum { ct_params, .. } => ct_params.is_some(),
            _ => false,
        };

        if type_is_assigned && !is_poly {
            return Ok(());
        }

        match self.parser.nodes[id] {
            Node::Symbol(sym) => {
                let resolved = self.scope_get(sym, id)?;
                self.assign_type(resolved)?;
                self.match_types(id, resolved);
                self.parser.node_is_addressable[id] = self.parser.node_is_addressable[resolved];
                Ok(())
            }
            Node::IntLiteral(_) => {
                self.types[id] = Type::Basic(BasicType::IntLiteral);
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
            Node::StringLiteral { .. } => {
                self.types[id] = Type::String;
                Ok(())
            }
            Node::Let { name: _, ty, expr } => {
                if let Some(expr) = expr {
                    self.assign_type(expr)?;
                    self.match_types(id, expr);
                }

                if let Some(ty) = ty {
                    self.assign_type(ty)?;

                    if let Some(poly_copy) = self.get_poly_copy(ty) {
                        self.match_types(id, poly_copy);
                    } else {
                        self.match_types(id, ty);
                    }
                }

                Ok(())
            }
            Node::Set { name, expr, .. } => {
                let saved_lhs_assign = self.lhs_assign;
                self.lhs_assign = true;
                self.assign_type(name)?;
                self.lhs_assign = saved_lhs_assign;

                self.assign_type(expr)?;

                self.match_types(id, expr);
                self.match_types(name, expr);

                // if name is a load, then we are doing a store
                let is_load = match &self.parser.nodes[name] {
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
            Node::If {
                cond,
                true_stmts,
                false_stmts
            } => {
                self.assign_type(cond)?;

                for stmt in self.parser.id_vec(true_stmts).clone() {
                    self.assign_type(stmt)?;
                }
                for stmt in self.parser.id_vec(false_stmts).clone() {
                    self.assign_type(stmt)?;
                }
                
                Ok(())
            }
            Node::Field {
                base,
                field_name,
                field_index: _,
                ..
            } => {
                self.assign_type(base)?;
                self.parser.node_is_addressable[base] = true;

                // todo(chad): deref more than once?
                let unpointered_ty = match &self.types[base] {
                    Type::Pointer(id) => *id,
                    _ => base,
                };

                if self.lhs_assign {
                    match self.parser.nodes.get_mut(id).unwrap() {
                        Node::Field { is_assignment, .. } => {
                            *is_assignment = true;
                        }
                        _ => unreachable!(),
                    }
                }

                match &self.types[unpointered_ty] {
                    Type::String => {
                        let field_name = self.parser.resolve_sym_unchecked(field_name);
                        match field_name {
                            "len" => {
                                self.types[id] = Type::Basic(BasicType::I64);
                            },
                            "buf" => {
                                // todo(chad): maybe make StringPointer its own type or something?
                                // or have Ids that will always reference basic types
                                self.types[id] = Type::Type;
                            },
                            _ => return Err(CompileError::from_string("Valid fields on strings are 'len' or 'buf'", self.parser.ranges[id])),
                        }

                        return Ok(())
                    }
                    _ => ()
                }

                let params = self.types[unpointered_ty].as_struct_params().ok_or(
                    CompileError::from_string(
                        format!(
                            "Expected struct or enum, got {:?} for node {}",
                            self.parser.debug(unpointered_ty),
                            unpointered_ty,
                        ),
                        self.parser.ranges[base],
                    ),
                )?;
                let params = self.get_struct_params(params);

                match params.get(&field_name) {
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
            Node::ArrayAccess { arr, index } => {
                // just like field accesses
                self.parser.node_is_addressable[id] = true;

                // todo(chad): auto-deref?
                self.assign_type(arr)?;
                self.assign_type(index)?;

                let array_ty = match self.types[arr] {
                    Type::Array(ty) => {
                        Ok(ty)
                    },
                    _ => Err(CompileError::from_string(format!("Cannot perform array access on non-array type {:?}", &self.types[id]), self.parser.ranges[id]))
                }?;
                self.match_types(id, array_ty);

                // todo(chad): assert that `index` has i64 type
                self.types[index] = Type::Basic(BasicType::I64);

                Ok(())
            }
            Node::StructLiteral { name: _, params } => {
                // todo(chad): @Performance this does not always need to be true, see comment in backend (compile_id on Node::StructLiteral)
                self.parser.node_is_addressable[id] = true;

                for param in self.parser.id_vec(params).clone() {
                    self.assign_type(param)?;
                }

                self.types[id] = Type::Struct {
                    params: params.clone(),
                    copied_from: None,
                };

                Ok(())
            }
            Node::ArrayLiteral(params) => {
                // todo(chad): @Performance this does not always need to be true, see comment in backend (compile_id on Node::StructLiteral)
                self.parser.node_is_addressable[id] = true;

                let params = self.parser.id_vec(params).clone();

                // todo(chad): Allow zero-element array literals?
                let matcher = params.first()
                    .cloned()
                    .ok_or_else(|| CompileError::from_string("Cannot have a zero-element array literal", self.parser.ranges[id]))?;
                for &param in params.iter() {
                    self.assign_type(param)?;
                    self.match_types(param, matcher);
                }

                self.types[id] = Type::Array(matcher);

                Ok(())
            }
            // Node::Add(arg1, arg2)
            // | Node::Sub(arg1, arg2)
            // | Node::Mul(arg1, arg2)
            // | Node::Div(arg1, arg2) => {
            //     self.assign_type(*arg1, coercion)?;
            //     self.assign_type(*arg2, coercion.or_id(*arg1))?;

            //     self.match_types(*arg1, *arg2);
            //     self.match_types(id, *arg1);

            //     Ok(())
            // }
            // Node::LessThan(arg1, arg2)
            // | Node::GreaterThan(arg1, arg2)
            // | Node::EqualTo(arg1, arg2) => {
            //     self.assign_type(*arg1, coercion)?;
            //     self.assign_type(*arg2, coercion.or_id(*arg1))?;

            //     self.match_types(*arg1, *arg2);
            //     self.match_types(id, *arg1);

            //     Ok(())
            // }
            // Node::And(arg1, arg2) | Node::Or(arg1, arg2) => {
            //     self.assign_type(*arg1, coercion)?;
            //     self.assign_type(*arg2, coercion.or_id(*arg1))?;

            //     Ok(())
            // }
            // Node::Not(arg1) => {
            //     self.assign_type(*arg1, Coercion::Basic(BasicType::Bool))?;
            //     Ok(())
            // }
            Node::Func {
                name: _,   // Sym,
                scope: _,  // Id,
                ct_params, // IdVec,
                params,    // IdVec,
                return_ty, // Id,
                stmts,     // IdVec,
                returns,   // IdVec,
                is_macro: _,  // bool,
            } => {
                if let Some(ct_params) = ct_params {
                    for ct_param in self.parser.id_vec(ct_params).clone() {
                        self.assign_type(ct_param)?;
                    }
                }

                self.assign_type(return_ty)?;

                for param in self.parser.id_vec(params).clone() {
                    self.assign_type(param)?;
                }

                self.types[id] = Type::Func {
                    return_ty: return_ty,
                    input_tys: params.clone(),
                    copied_from: None,
                };

                for stmt in self.parser.id_vec(stmts).clone() {
                    self.assign_type(stmt)?;
                    self.unify_types(false)?;
                }

                for ret_id in self.parser.id_vec(returns).clone() {
                    let ret_id = match self.parser.nodes[ret_id] {
                        Node::Return(id) => id,
                        _ => unreachable!(),
                    };

                    self.match_types(return_ty, ret_id);
                }

                self.topo.push(id);

                Ok(())
            }
            Node::Extern {
                name: _,
                params,
                return_ty,
            } => {
                for param in self.parser.id_vec(params).clone() {
                    self.assign_type(param)?;
                }
                self.assign_type(return_ty)?;

                self.types[id] = Type::Func {
                    return_ty,
                    input_tys: params,
                    copied_from: None
                };

                Ok(())
            }
            Node::DeclParam {
                name: _,  // Sym
                ty,       // Id
                index: _, // u16
                is_ct: _, // bool
                ct_link,
            } => {
                self.assign_type(ty)?;

                if let Some(ct_link) = ct_link {
                    self.assign_type(ct_link)?;
                    self.match_types(id, ct_link);
                }

                self.match_types(id, ty);
                Ok(())
            }
            Node::ValueParam { name: _, value, .. } => {
                self.assign_type(value)?;
                self.match_types(id, value);
                Ok(())
            }
            Node::Return(ret_id) => {
                self.assign_type(ret_id)?;
                self.match_types(id, ret_id);
                Ok(())
            }
            // Node::While(cond, stmts) => {
            //     self.assign_type(*cond, Coercion::Basic(BasicType::Bool))?;
            //     for &stmt in stmts {
            //         self.assign_type(stmt)?;
            //     }

            //     Ok(())
            // }
            Node::Ref(ref_id) => {
                // todo(chad): coercion
                self.assign_type(ref_id)?;
                self.types[id] = Type::Pointer(ref_id);

                Ok(())
            }
            Node::Load(load_id) => {
                // todo(chad): coercion
                self.assign_type(load_id)?;

                match self.types[load_id] {
                    Type::Pointer(pid) => {
                        self.match_types(id, pid);
                        Ok(())
                    }
                    _ => Err(CompileError::from_string(
                        "Cannot load a non-pointer",
                        self.parser.ranges[id],
                    )),
                }
            }
            Node::TypeLiteral(ty) => {
                self.types[id] = Type::Type;

                match ty {
                    Type::Basic(_) => {
                        self.types[id] = ty.clone();
                    }
                    Type::Pointer(pointer_ty) => {
                        self.assign_type(pointer_ty)?;
                        self.types[id] = ty.clone();
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
                        ..
                    } => {
                        self.assign_type(return_ty)?;
                        for input_ty in self.parser.id_vec(input_tys).clone() {
                            self.assign_type(input_ty)?;
                        }

                        self.types[id] = ty.clone();
                    }
                    Type::Struct { params, .. } => {
                        for ty in self.parser.id_vec(params).clone() {
                            self.assign_type(ty)?;
                        }

                        self.types[id] = ty.clone();
                    }
                    Type::Enum { params, .. } => {
                        for ty in self.parser.id_vec(params).clone() {
                            self.assign_type(ty)?;
                        }

                        self.types[id] = ty.clone();
                    }
                    Type::Array(array_ty) => {
                        self.assign_type(array_ty)?;
                        self.types[id] = ty.clone();
                    }
                    Type::Code | Type::Tokens => {
                        self.types[id] = ty.clone();
                    }
                    Type::Unassigned => unreachable!(),
                }

                Ok(())
            }
            Node::MacroCall {
                name,
                params,
                expanded,
            } => {
                self.topo.push(id);

                self.assign_type(name)?;

                let given = self.parser.id_vec(params).clone();
                let decl = match &self.parser.nodes[name] {
                    Node::Func { params, .. } => params,
                    Node::Extern { params, .. } => params,
                    _ => match &self.types[name] {
                        Type::Func { input_tys, .. } => input_tys,
                        _ => {
                            dbg!(&self.types[name]);
                            unreachable!();
                        }
                    },
                };
                let decl = self.parser.id_vec(*decl).clone();

                for (g, d) in given.iter().zip(decl.iter()) {
                    self.assign_type(*d)?;
                    self.assign_type(*g)?;
                    self.match_types(*d, *g);
                }

                for stmt in self.parser.id_vec(expanded).clone() {
                    self.assign_type(stmt)?;
                }

                Ok(())
            }
            Node::Call {
                name,
                ct_params,
                params,
            } => {
                let resolved_func = self.resolve(name)?;

                let is_ct = match &self.parser.nodes[resolved_func] {
                    Node::Func { ct_params, .. } => ct_params.is_some(),
                    Node::Extern { .. } => false,
                    _ => match &self.types[resolved_func] {
                        // if it's a pure function type (already specialized) then it's not ct
                        Type::Func { .. } => false,
                        // otherwise what the heck are we trying to call?
                        _ => {
                            return Err(CompileError::from_string(
                                &format!("Cannot call a non-func: {:?}", &self.parser.nodes[resolved_func]),
                                self.parser.ranges[id],
                            ))
                        }
                    },
                };

                let (name, resolved_func) = if is_ct {
                    let copied = self.deep_copy_fn(resolved_func);

                    // todo(chad): @Delete ??
                    match self.parser.nodes.get_mut(id).unwrap() {
                        Node::Call { name, .. } => {
                            *name = copied;
                        }
                        _ => unreachable!(),
                    };

                    (copied, copied)
                } else {
                    (name, resolved_func)
                };

                if is_ct {
                    // match given ct_params against declared ct_params
                    let given = ct_params
                        .map(|p| self.parser.id_vec(p).clone())
                        .unwrap_or(Vec::new());
                    let decl_id_vec = *match &self.parser.nodes[resolved_func] {
                        Node::Func { ct_params, .. } => ct_params,
                        _ => unreachable!(),
                    };
                    let decl = decl_id_vec
                        .map(|p| self.parser.id_vec(p).clone())
                        .unwrap_or(Vec::new());

                    for (index, (g, d)) in given.iter().zip(decl.iter()).enumerate() {
                        self.assign_type(*d)?;
                        self.assign_type(*g)?;
                        self.match_types(*g, *d);

                        match self.types[*d] {
                            Type::Type => (),
                            _ => match self.parser.nodes.get_mut(decl[index]).unwrap() {
                                Node::DeclParam { ct_link, .. } => {
                                    *ct_link = Some(*g);
                                },
                                _ => unreachable!()
                            }
                        }
                    }
                }

                let given = self.parser.id_vec(params).clone();
                let decl = match &self.parser.nodes[resolved_func] {
                    Node::Func { params, .. } => params,
                    Node::Extern { params, .. } => params,
                    _ => match &self.types[resolved_func] {
                        Type::Func { input_tys, .. } => input_tys,
                        _ => unreachable!(),
                    },
                };
                let decl = self.parser.id_vec(*decl).clone();

                for (g, d) in given.iter().zip(decl.iter()) {
                    self.assign_type(*d)?;
                    self.assign_type(*g)?;
                    self.match_types(*d, *g);
                }

                self.assign_type(name)?;
                match self.types[name] {
                    Type::Func { return_ty, .. } => self.match_types(id, return_ty),
                    _ => (),
                }

                Ok(())
            }
            Node::Struct {
                ct_params, params, ..
            } => {
                // for recursion purposes, give the struct a placeholder type before anything else
                self.types[id] = Type::Type;

                let (ct_params, params, copied) = if ct_params.is_some() {
                    let copied = self.deep_copy_struct(id);

                    match &self.parser.nodes[id] {
                        Node::Struct {
                            ct_params, params, ..
                        } => (*ct_params, *params, Some(copied)),
                        _ => unreachable!(),
                    }
                } else {
                    (ct_params, params, None)
                };

                if let Some(ct_params) = ct_params {
                    for param in self.parser.id_vec(ct_params).clone() {
                        self.assign_type(param)?;
                    }
                }
                for param in self.parser.id_vec(params).clone() {
                    self.assign_type(param)?;
                }
                self.types[id] = Type::Struct {
                    params: params.clone(),
                    copied_from: copied,
                };

                Ok(())
            }
            Node::Enum { name: _, params } => {
                for param in self.parser.id_vec(params).clone() {
                    self.assign_type(param)?;
                }
                self.types[id] = Type::Enum {
                    params: params.clone(),
                    copied_from: None,
                };
                Ok(())
            }
            // Node::Code
            Node::Unquote(unq) => {
                self.assign_type(unq)?;
                self.match_types(id, unq);

                Ok(())
            }
            Node::Insert { unquotes, .. } => {
                for unq in self.parser.id_vec(unquotes).clone() {
                    self.assign_type(unq)?;
                }

                Ok(())
            }
            Node::Tokens(_) => {
                self.types[id] = Type::Tokens;

                Ok(())
            }
            Node::Code(stmts) => {
                self.types[id] = Type::Code;

                if !self.macro_phase {
                    for stmt in self.parser.id_vec(stmts).clone() {
                        self.assign_type(stmt)?;
                    }
                }

                Ok(())
            }
            // Node::UnquotedCodeInsert(stmts) => {
            //     for stmt in self.parser.id_vec(stmts).clone() {
            //         self.assign_type(stmt)?;
            //     }

            //     // todo(chad): dangerous?
            //     if let Some(&stmt) = self.parser.id_vec(stmts).first() {
            //         self.match_types(id, stmt);
            //     }

            //     Ok(())
            // }
            // _ => Err(CompileError::from_string(
            //     format!(
            //         "Cannot coerce type for AST node {:?}",
            //         &self.parser.nodes[id]
            //     ),
            //     self.parser.ranges[id],
            // )),
        }
    }

    fn get_struct_params(&self, params: IdVec) -> BTreeMap<usize, (usize, u16)> {
        self
            .parser
            .id_vec(params)
            .iter()
            .enumerate()
            .map(|(_index, id)| {
                let (name, ty, index) = match &self.parser.nodes[*id] {
                    Node::DeclParam { name, ty, index, .. } => Some((*name, *ty, *index)),
                    Node::ValueParam {
                        name,
                        value: _,
                        index,
                        is_ct: false,
                    } => Some((name.unwrap(), *id, *index)),
                    _ => None,
                }
                .unwrap();

                (name, (ty, index))
            })
            .collect::<BTreeMap<_, _>>()
    }

    fn deep_copy_fn(&mut self, id: Id) -> Id {
        // println!("copying function!");

        let range = self.parser.ranges[id];
        let source =
            &self.parser.lexer.original_source[range.start.char_offset..range.end.char_offset];

        self.parser
            .lexer
            .update_source_for_copy(source, range.start);
        self.parser.copying = true;

        let copied = self.parser.parse_fn(range.start).unwrap();

        // Replace the name so we don't get collisions (simply append the copied node id since that's always unique)
        let old_name = match &self.parser.nodes[id] {
            Node::Func { name, .. } => self.parser.resolve_sym_unchecked(*name).to_string(),
            _ => unreachable!(),
        };
        let new_name = self
            .parser
            .lexer
            .string_interner
            .get_or_intern(format!("{}_{}", old_name, copied));
        match self.parser.nodes.get_mut(copied).unwrap() {
            Node::Func { name, .. } => {
                *name = new_name;
            }
            _ => unreachable!(),
        };

        while self.types.len() < self.parser.nodes.len() {
            self.types.push(Type::Unassigned);
        }

        copied
    }

    fn deep_copy_struct(&mut self, id: Id) -> Id {
        // println!("copying struct!");

        let range = self.parser.ranges[id];
        let source =
            &self.parser.lexer.original_source[range.start.char_offset..range.end.char_offset];

        self.parser
            .lexer
            .update_source_for_copy(source, range.start);
        self.parser.copying = true;

        let copied = self.parser.parse_struct(range.start).unwrap();

        while self.types.len() < self.parser.nodes.len() {
            self.types.push(Type::Unassigned);
        }

        copied
    }

    pub fn deep_copy_stmt(&mut self, scope: Id, id: Id) -> Id {
        // println!("copying stmt!");

        let is_expr = match &self.parser.nodes[id] {
            Node::IntLiteral(_) => true,
            Node::BoolLiteral(_) => true,
            Node::FloatLiteral(_) => true,
            Node::Symbol(_) => true,
            Node::Ref(_) => true,
            Node::Load(_) => true,
            Node::StructLiteral { .. } => true,
            _ => {
                println!("{:?} is not an expression", self.parser.nodes[id]);
                false
            }
        };

        let range = self.parser.ranges[id];
        let source =
            &self.parser.lexer.original_source[range.start.char_offset..range.end.char_offset];

        self.parser
            .lexer
            .update_source_for_copy(source, range.start);
        self.parser.top_scope = scope;
        self.parser.copying = true;

        let copied = if is_expr {
            self.parser.parse_expression().unwrap()
        } else {
            self.parser.parse_fn_stmt().unwrap()
        };

        while self.types.len() < self.parser.nodes.len() {
            self.types.push(Type::Unassigned);
        }

        copied
    }

    pub fn resolve(&self, id: Id) -> Result<Id, CompileError> {
        match &self.parser.nodes[id] {
            Node::Symbol(sym) => self.scope_get(*sym, id),
            _ => Ok(id),
        }
    }

    fn find_type_array_index(&mut self, id: Id) -> Option<usize> {
        for (index, m) in self.type_matches.iter().enumerate() {
            if m.contains(&id) {
                return Some(index);
            }
        }

        None
    }

    fn type_specificity(&self, id: Id) -> i16 {
        match &self.types[id] {
            Type::Unassigned | Type::Type => 0,
            Type::Basic(BasicType::IntLiteral) => 1,
            _ => 2,
        }
    }

    fn match_types(&mut self, ty1: Id, ty2: Id) {
        if ty1 == ty2 {
            return;
        }

        if self.types[ty1] == Type::Unassigned {
            self.types[ty1] = self.types[ty2];
        }
        if self.types[ty2] == Type::Unassigned {
            self.types[ty2] = self.types[ty1];
        }

        match (self.types[ty1], self.types[ty2]) {
            (
                Type::Func {
                    return_ty: return_ty1,
                    input_tys: input_tys1,
                    ..
                },
                Type::Func {
                    return_ty: return_ty2,
                    input_tys: input_tys2,
                    ..
                },
            ) => {
                let input_tys1 = self.parser.id_vec(input_tys1).clone();
                let input_tys2 = self.parser.id_vec(input_tys2).clone();

                self.match_types(return_ty1, return_ty2);
                for (it1, it2) in input_tys1.iter().zip(input_tys2.iter()) {
                    self.match_types(*it1, *it2);
                }
            }
            (Type::Struct { params: st1, .. }, Type::Struct { params: st2, .. }) => {
                let st1 = self.parser.id_vec(st1).clone();
                let st2 = self.parser.id_vec(st2).clone();

                for (ty1, ty2) in st1.iter().zip(st2.iter()) {
                    self.match_types(*ty1, *ty2);
                }
            }
            (Type::Struct { params: st, .. }, Type::Enum { params: et, .. }) => {
                self.match_struct_to_enum(st, et);
            }
            (Type::Enum { params: et, .. }, Type::Struct { params: st, .. }) => {
                self.match_struct_to_enum(st, et);
            }
            (Type::Pointer(pt1), Type::Pointer(pt2)) => {
                self.match_types(pt1, pt2);
            }
            (_, _) => (),
        }

        let id1 = self.find_type_array_index(ty1);
        let id2 = self.find_type_array_index(ty2);

        match (id1, id2) {
            (None, None) => self.type_matches.push(vec![ty1, ty2]),
            (Some(id), None) => self.type_matches[id].push(ty2),
            (None, Some(id)) => self.type_matches[id].push(ty1),
            (Some(id1), Some(id2)) if id1 != id2 => {
                let lower = id1.min(id2);
                let upper = id1.max(id2);

                // todo(chad): partition or something
                let upper_matches = self.type_matches[upper].clone();
                self.type_matches[lower].extend(upper_matches);
                self.type_matches.remove(upper);
            }
            (_, _) => (),
        }
    }

    fn match_struct_to_enum(&mut self, st: IdVec, et: IdVec) {
        let struct_params = self.get_struct_params(st);
        let enum_params = self.get_struct_params(et);

        let st = self.parser.id_vec(st).clone();
        assert_eq!(st.len(), 1);

        let (name, (st, _)) = struct_params.iter().next().unwrap();
        let (et, _) = enum_params.get(name).unwrap();

        self.match_types(*st, *et);
    }

    fn get_poly_copy(&self, id: Id) -> Option<Id> {
        match self.types[id] {
            Type::Struct { copied_from, .. } => copied_from,
            Type::Enum { copied_from, .. } => copied_from,
            Type::Func { copied_from, .. } => copied_from,
            _ => None,
        }
    }

    pub fn unify_types(&mut self, final_pass: bool) -> Result<(), CompileError> {
        let mut to_clear = Vec::new();

        for uid in 0..self.type_matches.len() {
            // println!("\n********* Unifying *********");
            let tys = self.type_matches[uid]
                .iter()
                .map(|&ty| {
                    // println!(
                    //     "{:?} ({}) : {:?}",
                    //     // &self.parser.nodes[ty],
                    //     ty,
                    //     self.parser.debug(ty),
                    //     &self.types[ty],
                    // );
                    (ty, self.type_specificity(ty))
                })
                .filter(|(_, spec)| *spec > 0)
                .filter(|(ty, _)| {
                    if final_pass {
                        self.types[*ty as usize].is_concrete()
                    } else {
                        self.types[*ty as usize].is_coercble()
                    }
                })
                .map(|(ty, _)| ty)
                .collect::<Vec<_>>();

            // set all types to whatever we unified to
            if let Some(&ty) = tys.last() {
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

            for &ty in tys.iter() {
                let check_err = match (self.types[ty], self.types[tys[0]]) {
                    (Type::Basic(_), _) => true,
                    (_, Type::Basic(_)) => true,
                    _ => false,
                };

                if check_err && self.types[ty] != self.types[tys[0]] {
                    return Err(CompileError::from_string(
                        format!(
                            "Type unification failed for types {:?} and {:?}",
                            &self.types[ty], &self.types[tys[0]]
                        ),
                        self.parser.ranges[ty],
                    ));
                }
            }
        }

        to_clear.reverse();
        for uid in to_clear {
            self.type_matches.remove(uid);
        }

        Ok(())
    }

    pub fn scope_get(&self, sym: Sym, id: Id) -> Result<Id, CompileError> {
        self.parser.scope_get(sym, id).ok_or_else(|| {
            CompileError::from_string(
                format!(
                    "Undeclared identifier {}",
                    self.parser.resolve_sym_unchecked(sym)
                ),
                self.parser.ranges[id],
            )
        })
    }
}
