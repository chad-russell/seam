use crate::parser::{BasicType, CompileError, Id, IdVec, Node, Parser, Type};

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

    //         backend.semantic.unify_types()?;

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
            Node::Enum { ct_params, .. } => ct_params.is_some(),
            _ => false,
        };

        if type_is_assigned && !is_poly {
            return Ok(());
        }

        match self.parser.nodes[id] {
            Node::Symbol(sym) => {
                let resolved = self.scope_get(sym, id)?;
                self.assign_type(resolved)?;
                self.match_types(id, resolved)?;
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
                    self.match_types(id, expr)?;
                }

                if let Some(ty) = ty {
                    self.assign_type(ty)?;

                    if let Some(poly_copy) = self.get_poly_copy(ty) {
                        self.match_types(id, poly_copy)?;
                    } else {
                        self.match_types(id, ty)?;
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

                self.match_types(id, expr)?;
                self.match_types(name, expr)?;

                // if name is a load, then we are doing a store
                let is_load = match &self.parser.nodes[name] {
                    Node::Load(_) => true,
                    _ => false,
                };

                if is_load {
                    if let Node::Set { is_store, .. } = &mut self.parser.nodes[id] { *is_store = true; }
                }

                Ok(())
            }
            Node::If {
                cond,
                true_stmts,
                false_stmts,
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
            Node::While {
                cond,
                stmts,
            } => {
                self.assign_type(cond)?;

                for stmt in self.parser.id_vec(stmts).clone() {
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

                if self.types[unpointered_ty] == Type::String {
                    let field_name = self.parser.resolve_sym_unchecked(field_name);
                    match field_name {
                        "len" => {
                            self.types[id] = Type::Basic(BasicType::I64);
                        }
                        "ptr" => {
                            // todo(chad): maybe make StringPointer its own type or something?
                            // or have Ids that will always reference basic types
                            self.types[id] = Type::Type;
                        }
                        _ => {
                            return Err(CompileError::from_string(
                                "Valid fields on strings are 'len' or 'buf'",
                                self.parser.ranges[id],
                            ))
                        }
                    }

                    return Ok(());
                }

                if let Type::Array(_) = self.types[unpointered_ty] {
                    let field_name = self.parser.resolve_sym_unchecked(field_name);
                    match field_name {
                        "len" => {
                            self.types[id] = Type::Basic(BasicType::I64);
                        }
                        "ptr" => {
                            // todo(chad): maybe make StringPointer its own type or something?
                            // or have Ids that will always reference basic types
                            self.types[id] = Type::Type;
                        }
                        _ => {
                            return Err(CompileError::from_string(
                                "Valid fields on strings are 'len' or 'buf'",
                                self.parser.ranges[id],
                            ))
                        }
                    }

                    return Ok(());
                }

                let params = self.types[unpointered_ty].as_struct_params().ok_or_else(||
                    CompileError::from_string(
                        format!(
                            "Expected struct or enum, got {:?} for node {}",
                            // self.parser.debug(unpointered_ty),
                            self.types[unpointered_ty],
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

                        self.match_types(id, *ty)?;
                        Ok(())
                    }
                    _ => Err(CompileError::from_string(
                        format!(
                            "field '{}' not found: fields are {:?}",
                            self.parser
                                .lexer
                                .string_interner
                                .resolve(field_name)
                                .unwrap(),
                            params
                                .iter()
                                .map(|(sym, _)| self
                                    .parser
                                    .lexer
                                    .string_interner
                                    .resolve(*sym)
                                    .unwrap())
                                .collect::<Vec<_>>(),
                        ),
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
                    Type::Array(ty) => Ok(ty),
                    _ => Err(CompileError::from_string(
                        format!(
                            "Cannot perform array access on non-array type {:?}",
                            &self.types[id]
                        ),
                        self.parser.ranges[id],
                    )),
                }?;
                self.match_types(id, array_ty)?;

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

                self.types[id] = Type::StructLiteral(params);

                Ok(())
            }
            Node::ArrayLiteral(params) => {
                // todo(chad): @Performance this does not always need to be true, see comment in backend (compile_id on Node::StructLiteral)
                self.parser.node_is_addressable[id] = true;

                let params = self.parser.id_vec(params).clone();

                // todo(chad): Allow zero-element array literals?
                let matcher = params.first().cloned().ok_or_else(|| {
                    CompileError::from_string(
                        "Cannot have a zero-element array literal",
                        self.parser.ranges[id],
                    )
                })?;
                for &param in params.iter() {
                    self.assign_type(param)?;
                    if param != matcher {
                        self.match_types(param, matcher)?;
                    }
                }

                self.types[id] = Type::Array(matcher);

                Ok(())
            }
            Node::EqEq(lhs, rhs) | Node::Neq(lhs, rhs) => {
                self.assign_type(lhs)?;
                self.assign_type(rhs)?;
                self.match_types(lhs, rhs)?;

                self.types[id] = Type::Basic(BasicType::Bool);

                Ok(())
            }
            Node::Add(lhs, rhs) | Node::Sub(lhs, rhs) | Node::Mul(lhs, rhs) | Node::Div(lhs, rhs) | Node::LessThan(lhs, rhs) | Node::GreaterThan(lhs, rhs) => {
                self.assign_type(lhs)?;
                self.assign_type(rhs)?;
                self.match_types(lhs, rhs)?;
                self.match_types(id, lhs)?;

                Ok(())
            }
            Node::Func {
                name: _,     // Sym,
                scope: _,    // Id,
                ct_params,   // IdVec,
                params,      // IdVec,
                return_ty,   // Id,
                stmts,       // IdVec,
                returns,     // IdVec,
                is_macro: _, // bool,
                copied_from: _,
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
                    return_ty,
                    input_tys: params,
                    copied_from: None,
                };

                for stmt in self.parser.id_vec(stmts).clone() {
                    self.assign_type(stmt)?;
                    self.unify_types()?;
                }

                for ret_id in self.parser.id_vec(returns).clone() {
                    let ret_id = match self.parser.nodes[ret_id] {
                        Node::Return(id) => id,
                        _ => unreachable!(),
                    };

                    self.match_types(return_ty, ret_id)?;
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
                    copied_from: None,
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
                    self.match_types(id, ct_link)?;
                }

                self.match_types(id, ty)?;
                Ok(())
            }
            Node::ValueParam { name: _, value, .. } => {
                self.assign_type(value)?;
                self.match_types(id, value)?;
                Ok(())
            }
            Node::Return(ret_id) => {
                self.assign_type(ret_id)?;
                self.match_types(id, ret_id)?;
                Ok(())
            }
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
                        self.match_types(id, pid)?;
                        Ok(())
                    }
                    _ => Err(CompileError::from_string(
                        format!("Cannot load a non-pointer {:?}", self.types[load_id]),
                        self.parser.ranges[id],
                    )),
                }
            }
            Node::TypeLiteral(ty) => {
                self.types[id] = Type::Type;

                match ty {
                    Type::Basic(_) => {
                        self.types[id] = ty;
                    }
                    Type::Pointer(pointer_ty) => {
                        self.assign_type(pointer_ty)?;
                        self.types[id] = ty;
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

                        self.types[id] = ty;
                    }
                    Type::Struct { params, .. } => {
                        for ty in self.parser.id_vec(params).clone() {
                            self.assign_type(ty)?;
                        }
                        self.types[id] = ty;
                    }
                    Type::StructLiteral(params) => {
                        for ty in self.parser.id_vec(params).clone() {
                            self.assign_type(ty)?;
                        }
                        self.types[id] = ty;
                    }
                    Type::Enum { params, .. } => {
                        for ty in self.parser.id_vec(params).clone() {
                            self.assign_type(ty)?;
                        }

                        self.types[id] = ty;
                    }
                    Type::Array(array_ty) => {
                        self.assign_type(array_ty)?;
                        self.types[id] = ty;
                    }
                    Type::Tokens => {
                        self.types[id] = ty;
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
                self.assign_type(name)?;

                let given = self.parser.id_vec(params).clone();
                let decl = match &self.parser.nodes[name] {
                    Node::Func { params, .. } => params,
                    Node::Extern { params, .. } => params,
                    _ => match &self.types[name] {
                        Type::Func { input_tys, .. } => input_tys,
                        _ => unreachable!(),
                    },
                };
                let decl = self.parser.id_vec(*decl).clone();

                for (g, d) in given.iter().zip(decl.iter()) {
                    self.assign_type(*d)?;
                    self.assign_type(*g)?;
                    self.match_types(*d, *g)?;
                }

                self.types[id] = Type::Type;

                for stmt in self.parser.id_vec(expanded).clone() {
                    self.assign_type(stmt)?;
                    self.types[id] = self.types[stmt];
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
                                &format!(
                                    "Cannot call a non-func: {:?}",
                                    &self.parser.nodes[resolved_func]
                                ),
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
                        .unwrap_or_default();
                    let decl_id_vec = *match &self.parser.nodes[resolved_func] {
                        Node::Func { ct_params, .. } => ct_params,
                        _ => unreachable!(),
                    };
                    let decl = decl_id_vec
                        .map(|p| self.parser.id_vec(p).clone())
                        .unwrap_or_default();

                    for (index, (g, d)) in given.iter().zip(decl.iter()).enumerate() {
                        self.assign_type(*d)?;
                        self.assign_type(*g)?;
                        self.match_types(*g, *d)?;

                        match self.types[*d] {
                            Type::Type => (),
                            _ => match self.parser.nodes.get_mut(decl[index]).unwrap() {
                                Node::DeclParam { ct_link, .. } => {
                                    *ct_link = Some(*g);
                                }
                                _ => unreachable!(),
                            },
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
                    self.match_types(*d, *g)?;
                }

                self.unify_types()?;

                self.assign_type(name)?;
                if let Type::Func { return_ty, .. } = self.types[name] {
                    self.match_types(id, return_ty)?;
                }

                Ok(())
            }
            Node::Struct {
                ct_params, params, ..
            } => {
                // for recursion purposes, give the struct a placeholder type before anything else
                self.types[id] = Type::Type;

                let (ct_params, params) = if ct_params.is_some() {
                    let copied = self.deep_copy_struct(id);

                    let copied_params = match self.parser.nodes[copied] {
                        Node::Struct { params, .. } => params,
                        _ => unreachable!(),
                    };

                    self.types[copied] = Type::Struct {
                        params: copied_params,
                        copied_from: None,
                    };
                    self.match_types(id, copied)?;

                    match &self.parser.nodes[copied] {
                        Node::Struct {
                            ct_params, params, ..
                        } => (*ct_params, *params),
                        _ => unreachable!(),
                    }
                } else {
                    self.types[id] = Type::Struct {
                        params,
                        copied_from: None,
                    };

                    (ct_params, params)
                };

                if let Some(ct_params) = ct_params {
                    for param in self.parser.id_vec(ct_params).clone() {
                        self.assign_type(param)?;
                    }
                }
                for param in self.parser.id_vec(params).clone() {
                    self.assign_type(param)?;
                }

                Ok(())
            }
            Node::Enum {
                ct_params, params, ..
            } => {
                // for recursion purposes, give the struct a placeholder type before anything else
                self.types[id] = Type::Type;

                let (ct_params, params) = if ct_params.is_some() {
                    let copied = self.deep_copy_enum(id);

                    let copied_params = match self.parser.nodes[copied] {
                        Node::Enum { params, .. } => params,
                        _ => unreachable!(),
                    };

                    self.types[copied] = Type::Enum {
                        params: copied_params,
                        copied_from: None,
                    };
                    self.match_types(id, copied)?;

                    match &self.parser.nodes[copied] {
                        Node::Enum {
                            ct_params, params, ..
                        } => (*ct_params, *params),
                        _ => unreachable!(),
                    }
                } else {
                    self.types[id] = Type::Enum {
                        params,
                        copied_from: None,
                    };

                    (ct_params, params)
                };

                if let Some(ct_params) = ct_params {
                    for param in self.parser.id_vec(ct_params).clone() {
                        self.assign_type(param)?;
                    }
                }
                for param in self.parser.id_vec(params).clone() {
                    self.assign_type(param)?;
                }

                Ok(())
            }
            Node::TypeOf(expr) => {
                self.types[id] = Type::Type;

                self.assign_type(expr)?;
                self.match_types(id, self.parser.ty_decl.unwrap())?;

                Ok(())
            }
            Node::Cast(expr) => {
                self.assign_type(expr)?;
                self.parser.node_is_addressable[id] = self.parser.node_is_addressable[expr];

                Ok(())
            }
            Node::Tokens(_) => {
                self.types[id] = Type::Tokens;

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

    fn get_struct_params(&self, params: IdVec) -> BTreeMap<usize, (usize, u16)> {
        self.parser
            .id_vec(params)
            .iter()
            .enumerate()
            .map(|(_index, id)| {
                let (name, ty, index) = match &self.parser.nodes[*id] {
                    Node::DeclParam {
                        name, ty, index, ..
                    } => Some((*name, *ty, *index)),
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
        // println!("deep copying function");

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
            Node::Func {
                name, copied_from, ..
            } => {
                *name = new_name;
                *copied_from = Some(id);
            }
            _ => unreachable!(),
        };

        while self.types.len() < self.parser.nodes.len() {
            self.types.push(Type::Unassigned);
        }

        copied
    }

    fn deep_copy_struct(&mut self, id: Id) -> Id {
        println!("copying struct!");

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

    fn deep_copy_enum(&mut self, id: Id) -> Id {
        println!("copying enum!");

        let range = self.parser.ranges[id];
        let source =
            &self.parser.lexer.original_source[range.start.char_offset..range.end.char_offset];

        self.parser
            .lexer
            .update_source_for_copy(source, range.start);
        self.parser.copying = true;

        let copied = self.parser.parse_enum(range.start).unwrap();

        while self.types.len() < self.parser.nodes.len() {
            self.types.push(Type::Unassigned);
        }

        copied
    }

    #[allow(dead_code)]
    pub fn deep_copy_stmt(&mut self, scope: Id, id: Id) -> Id {
        println!("deep copying stmt!");

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

    fn check_int_literal_type(&mut self, bt: BasicType) {
        if bt == BasicType::None || bt == BasicType::Bool {
            todo!("non-matching int literal type");
        }
    }

    fn match_types(&mut self, ty1: Id, ty2: Id) -> Result<(), CompileError> {
        if ty1 == ty2 {
            return Ok(());
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

                self.match_types(return_ty1, return_ty2)?;
                for (it1, it2) in input_tys1.iter().zip(input_tys2.iter()) {
                    self.match_types(*it1, *it2)?;
                }
            }
            (Type::Struct { params: st1, .. }, Type::Struct { params: st2, .. }) => {
                let st1 = self.parser.id_vec(st1).clone();
                let st2 = self.parser.id_vec(st2).clone();

                for (ty1, ty2) in st1.iter().zip(st2.iter()) {
                    self.match_types(*ty1, *ty2)?;
                }
            }
            (Type::StructLiteral(st1), Type::Struct { params: st2, .. }) => {
                let st1 = self.parser.id_vec(st1).clone();
                let st2 = self.parser.id_vec(st2).clone();

                for (ty1, ty2) in st1.iter().zip(st2.iter()) {
                    self.match_types(*ty1, *ty2)?;
                }
            }
            (Type::Struct { params: st1, .. }, Type::StructLiteral(st2)) => {
                let st1 = self.parser.id_vec(st1).clone();
                let st2 = self.parser.id_vec(st2).clone();

                for (ty1, ty2) in st1.iter().zip(st2.iter()) {
                    self.match_types(*ty1, *ty2)?;
                }
            }
            (Type::Enum { params: et1, .. }, Type::Enum { params: et2, .. }) => {
                let et1 = self.parser.id_vec(et1).clone();
                let et2 = self.parser.id_vec(et2).clone();

                for (ty1, ty2) in et1.iter().zip(et2.iter()) {
                    self.match_types(*ty1, *ty2)?;
                }
            }
            (Type::StructLiteral(st), Type::Enum { params: et, .. }) => {
                self.match_struct_to_enum(st, et)?;
            }
            (Type::Enum { params: et, .. }, Type::StructLiteral(st)) => {
                self.match_struct_to_enum(st, et)?;
            }
            (Type::Pointer(pt1), Type::Pointer(pt2)) => {
                self.match_types(pt1, pt2)?;
            }
            (Type::Array(at1), Type::Array(at2)) => {
                self.match_types(at1, at2)?;
            }
            (Type::Tokens, Type::Tokens) => (),
            (Type::String, Type::String) => (),
            (Type::Basic(bt1), Type::Basic(bt2)) if bt1 == bt2 => (),
            (Type::Basic(BasicType::IntLiteral), Type::Basic(bt)) => {
                self.check_int_literal_type(bt)
            }
            (Type::Basic(bt), Type::Basic(BasicType::IntLiteral)) => {
                self.check_int_literal_type(bt)
            }
            (Type::StructLiteral(_), Type::StructLiteral(_)) => (), // could be two enum variants in which case they wouldn't appear to match
            (Type::Type, _) => (),
            (_, Type::Type) => (),
            (Type::Unassigned, _) => (),
            (_, Type::Unassigned) => (),
            (_, _) => {
                return Err(CompileError::from_string(
                    format!(
                        "type mismatch: {:?} vs {:?} (ty2 is at {:?})",
                        self.types[ty1], self.types[ty2], self.parser.ranges[ty2]
                    ),
                    self.parser.ranges[ty1],
                ))
            }
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

        Ok(())
    }

    // todo(chad): only should match enum to struct if it's a struct literal -- NOT a real struct
    fn match_struct_to_enum(&mut self, st: IdVec, et: IdVec) -> Result<(), CompileError> {
        let struct_params = self.get_struct_params(st);
        let enum_params = self.get_struct_params(et);

        let st = self.parser.id_vec(st).clone();
        assert_eq!(st.len(), 1);

        let (name, (st, _)) = struct_params.iter().next().unwrap();
        let (et, _) = enum_params.get(name).unwrap();

        self.match_types(*st, *et)
    }

    fn get_poly_copy(&self, id: Id) -> Option<Id> {
        match self.types[id] {
            Type::Struct { copied_from, .. } => copied_from,
            Type::Enum { copied_from, .. } => copied_from,
            Type::Func { copied_from, .. } => copied_from,
            _ => None,
        }
    }

    pub fn is_concrete(&self, id: Id) -> bool {
        match self.types[id] {
            Type::Type
            | Type::Unassigned
            | Type::Basic(BasicType::IntLiteral)
            | Type::StructLiteral(_) => false,
            Type::Array(ty) => self.is_concrete(ty),
            _ => true,
        }
    }

    pub fn unify_types(&mut self) -> Result<(), CompileError> {
        // as long as we're clearing stuff out, keep going
        while self.unify_types_internal()? {}

        Ok(())
    }

    pub fn unify_types_internal(&mut self) -> Result<bool, CompileError> {
        let mut to_clear = Vec::new();

        let mut future_matches = Vec::new();
        let mut future_enum_matches = Vec::new();
        // todo(chad): @Performance
        for uid in 0..self.type_matches.len() {
            // println!("\n********* Unifying *********");

            let tys = self.type_matches[uid]
                .iter()
                .map(|&ty| {
                    // println!(
                    //     "{:?} ({}) : {:?}",
                    //     ty,
                    //     self.parser.debug(ty),
                    //     &self.types[ty],
                    // );
                    (ty, self.type_specificity(ty))
                })
                .filter(|(_, spec)| *spec > 0)
                .filter(|(ty, _)| self.is_concrete(*ty))
                .map(|(ty, _)| ty)
                .collect::<Vec<_>>();

            // set all types to whatever we unified to
            if let Some(&ty) = tys.first() {
                to_clear.push(uid);

                // println!(
                //     "setting all to {:?} (from {:?})",
                //     self.types[ty as usize],
                //     self.parser.debug(ty as _)
                // );

                for id in self.type_matches[uid].clone() {
                    // if setting a struct literal to a struct/enum, do more matching on the fields
                    if let Type::StructLiteral(lit_params) = self.types[id] {
                        match self.types[ty as usize] {
                            Type::Struct { params, .. } => {
                                // todo(chad): make this a method
                                // todo(chad): check they have the same number of arguments
                                let st1 = self.parser.id_vec(lit_params).clone();
                                let st2 = self.parser.id_vec(params).clone();

                                for (ty1, ty2) in st1.iter().zip(st2.iter()) {
                                    future_matches.push((*ty1, *ty2));
                                }
                            }
                            Type::Enum { params, .. } => {
                                future_enum_matches.push((lit_params, params));
                            }
                            _ => unreachable!(),
                        }
                    }

                    self.types[id] = self.types[ty as usize];
                }
            }
        }

        let made_progress = !to_clear.is_empty();

        to_clear.reverse();
        for uid in to_clear {
            self.type_matches.remove(uid);
        }

        for (id1, id2) in future_matches {
            self.match_types(id1, id2)?;
        }
        for (lit_params, params) in future_enum_matches {
            self.match_struct_to_enum(lit_params, params)?;
        }

        Ok(made_progress)
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

    pub fn allocate_for_new_nodes(&mut self) {
        while self.types.len() < self.parser.nodes.len() {
            self.types.push(Type::Unassigned);
        }
    }
}
