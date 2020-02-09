use std::collections::BTreeMap;
use std::convert::TryInto;
use std::ops::Deref;

use string_interner::StringInterner;

type Sym = usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Location {
    line: usize,
    col: usize,
}

impl Default for Location {
    fn default() -> Self {
        Self { line: 1, col: 0 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Range {
    start: Location,
    end: Location,
}

impl Range {
    pub fn new(start: Location, end: Location) -> Self {
        Self { start, end }
    }
}

impl Into<Range> for Location {
    fn into(self) -> Range {
        Range::new(self, self)
    }
}

pub type Id = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasicType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    None,
}

// TODO: make this copy
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Unassigned,
    Basic(BasicType),
    Pointer(Id),
    String,
    Func { return_ty: Id, input_tys: Vec<Id> },
    Struct(Vec<(Id, Sym)>),
}

impl Type {
    pub fn as_struct_params(&self) -> Option<&Vec<(Id, Sym)>> {
        match self {
            Type::Struct(params) => Some(params),
            _ => None,
        }
    }
}

impl Into<Type> for BasicType {
    fn into(self) -> Type {
        Type::Basic(self)
    }
}

// todo(chad): make this copy
#[derive(Debug, Clone, PartialEq)]
enum Token {
    LParen,
    RParen,
    LCurly,
    RCurly,
    Semicolon,
    Colon,
    Dot,
    Asterisk,
    Eq,
    Comma,
    True,
    False,
    Ampersand,
    Caret,
    Uninit,
    Symbol(Sym),
    IntegerLiteral(i64), // TODO: handle negative literals
    FloatLiteral(f64),   // TODO: handle negative literals
    StringLiteral(String),
    EOF,
}

#[derive(Debug, Clone, PartialEq)]
struct Lexeme {
    tok: Token,
    range: Range,
}

impl Lexeme {
    fn new(tok: Token, range: Range) -> Self {
        Self { tok, range }
    }
}

pub struct Lexer<'a> {
    pub string_interner: Box<StringInterner<Sym>>,

    pub source: &'a str,
    pub loc: Location,

    top: Lexeme,
    second: Lexeme,
}

fn is_special(c: char) -> bool {
    c == ' '
        || c == '\t'
        || c == '\n'
        || c == '\r'
        || c == '{'
        || c == '}'
        || c == '('
        || c == ')'
        || c == '['
        || c == ']'
        || c == '+'
        || c == '-'
        || c == '*'
        || c == '/'
        || c == '-'
        || c == '.'
        || c == ':'
        || c == '\''
        || c == '"'
        || c == '`'
        || c == '!'
        || c == '|'
        || c == ','
        || c == ';'
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        let string_interner = Box::new(StringInterner::new());

        Self {
            string_interner,
            source,
            loc: Default::default(),
            top: Lexeme::new(Token::EOF, Default::default()),
            second: Lexeme::new(Token::EOF, Default::default()),
        }
    }

    pub fn update_source(&mut self, source: &'a str) {
        self.source = source;
        self.loc = Default::default();
        self.top = Lexeme::new(Token::EOF, Default::default());
        self.second = Lexeme::new(Token::EOF, Default::default());

        self.pop();
        self.pop();
    }

    #[inline]
    fn eat(&mut self, chars: usize) {
        self.source = &self.source[chars..];
        self.loc.col += chars;
    }

    #[inline]
    fn newline(&mut self) {
        self.source = &self.source[1..];
        self.loc.line += 1;
        self.loc.col = 1;
    }

    #[inline]
    fn eat_rest_of_line(&mut self) {
        let chars = self
            .source
            .chars()
            .position(|c| c == '\n')
            .unwrap_or(self.source.len());
        self.eat(chars);
        self.newline();
    }

    #[inline]
    fn eat_spaces(&mut self) {
        loop {
            let mut br = true;
            while let Some(' ') = self.source.chars().next() {
                br = false;
                self.eat(1);
            }
            while let Some('\r') = self.source.chars().next() {
                br = false;
                self.newline();
            }
            while let Some('\n') = self.source.chars().next() {
                br = false;
                self.newline();
            }
            while self.source.len() > 1 && &self.source[0..2] == "//" {
                br = false;
                self.eat_rest_of_line();
            }

            if br {
                break;
            }
        }
    }

    fn resolve_unchecked(&self, sym: Sym) -> &str {
        self.string_interner.deref().resolve(sym).unwrap()
    }

    fn prefix(&mut self, pat: &str, tok: Token) -> bool {
        if self.source.len() >= pat.len() + 1 && self.source.starts_with(pat) {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, Range::new(start, self.loc));
            true
        } else {
            false
        }
    }

    fn prefix_keyword(&mut self, pat: &str, tok: Token) -> bool {
        if self.source.len() >= pat.len() + 1
            && self.source.starts_with(pat)
            && is_special(self.source.chars().skip(pat.len()).take(1).next().unwrap())
        {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, Range::new(start, self.loc));
            true
        } else {
            false
        }
    }

    pub fn pop(&mut self) {
        self.eat_spaces();

        let start = self.loc;

        self.top = self.second.clone();

        if self.prefix_keyword("---", Token::Uninit) {
            return;
        }
        if self.prefix_keyword("true", Token::True) {
            return;
        }
        if self.prefix_keyword("false", Token::False) {
            return;
        }

        if self.prefix("(", Token::LParen) {
            return;
        }
        if self.prefix(")", Token::RParen) {
            return;
        }
        if self.prefix("{", Token::LCurly) {
            return;
        }
        if self.prefix("}", Token::RCurly) {
            return;
        }
        if self.prefix(",", Token::Comma) {
            return;
        }
        if self.prefix("&", Token::Ampersand) {
            return;
        }
        if self.prefix(".", Token::Dot) {
            return;
        }
        if self.prefix("^", Token::Caret) {
            return;
        }
        if self.prefix(";", Token::Semicolon) {
            return;
        }
        if self.prefix("*", Token::Asterisk) {
            return;
        }
        if self.prefix(":", Token::Colon) {
            return;
        }
        if self.prefix("=", Token::Eq) {
            return;
        }

        self.second = match self.source.chars().next() {
            Some('"') => {
                self.eat(1);
                let index = match self.source.chars().position(|c| c == '"') {
                    Some(index) => index,
                    None => self.source.len(),
                };

                let s = String::from(&self.source[0..index]);

                self.eat(index + 1);
                Lexeme::new(Token::StringLiteral(s), Range::new(start, self.loc))
            }
            Some(c) if c.is_digit(10) => {
                let index = match self.source.chars().position(|c| !c.is_digit(10)) {
                    Some(index) => index,
                    None => self.source.len(),
                };

                let has_decimal = match self.source.get(index..index + 1) {
                    Some(c) => c == ".",
                    _ => false,
                };

                let digit = self.source[..index]
                    .parse::<i64>()
                    .expect("Failed to parse numeric literal");

                self.eat(index);

                if has_decimal {
                    self.eat(1);

                    let decimal_index = match self.source.chars().position(|c| !c.is_digit(10)) {
                        Some(index) => index,
                        None => self.source.len(),
                    };

                    let decimal_digit = self.source[..decimal_index]
                        .parse::<i64>()
                        .expect("Failed to parse numeric literal");

                    self.eat(decimal_index);

                    let digit: f64 = format!("{}.{}", digit, decimal_digit).parse().unwrap();

                    let end = self.loc;
                    Lexeme::new(Token::FloatLiteral(digit), Range::new(start, end))
                } else {
                    let end = self.loc;
                    Lexeme::new(Token::IntegerLiteral(digit), Range::new(start, end))
                }
            }
            Some(_) => {
                let index = match self.source.chars().position(is_special) {
                    Some(index) => index,
                    None => self.source.len(),
                };

                let sym = self.string_interner.get_or_intern(&self.source[..index]);
                self.eat(index);

                let end = self.loc;

                Lexeme::new(Token::Symbol(sym), Range::new(start, end))
            }
            None => Lexeme::new(Token::EOF, Default::default()),
        };
    }
}

#[derive(Default, Debug)]
pub struct Scope {
    pub parent: Option<Id>,
    pub entries: BTreeMap<Sym, Id>,
    pub locals: Vec<Id>,
}

impl Scope {
    fn new(parent: Id) -> Self {
        Self {
            parent: Some(parent),
            entries: Default::default(),
            locals: Default::default(),
        }
    }
}

pub struct Parser<'a> {
    pub nodes: Vec<Node>,
    pub ranges: Vec<Range>,

    pub node_is_addressable: Vec<bool>,

    pub lexer: Lexer<'a>,

    pub node_scopes: Vec<Id>,

    pub scopes: Vec<Scope>,
    top_scope: Id,

    // todo(chad): represent list of locals per function
    pub top_level: Vec<Id>,
    pub top_level_map: BTreeMap<Sym, Id>,

    sym_fn: Sym,
    sym_struct: Sym,
    sym_field: Sym,
    sym_let: Sym,
    sym_uninit: Sym,
    sym_set: Sym,
    sym_store: Sym,
    sym_ptr: Sym,
    sym_ref: Sym,
    sym_load: Sym,
    sym_ptrcast: Sym,
    sym_call: Sym,
    sym_return: Sym,
    sym_none: Sym,
    sym_i8: Sym,
    sym_i64: Sym,
    sym_i32: Sym,
    sym_i16: Sym,
    sym_f32: Sym,
    sym_f64: Sym,
    sym_bool: Sym,
    sym_string: Sym,
    sym_plus: Sym,
    sym_minus: Sym,
    sym_mul: Sym,
    sym_div: Sym,
    sym_lt: Sym,
    sym_gt: Sym,
    sym_eq: Sym,
    sym_and: Sym,
    sym_or: Sym,
    sym_not: Sym,
    sym_if: Sym,
    sym_while: Sym,
    sym_extern: Sym,
}

#[derive(Debug, Clone)]
pub struct CompileErrorNote {
    pub msg: String,
    pub range: Range,
}

#[derive(Debug, Clone)]
pub struct CompileError {
    pub note: CompileErrorNote,
    pub extra_notes: Vec<CompileErrorNote>,
}

impl CompileError {
    pub fn from_string(msg: impl Into<String>, range: impl Into<Range>) -> Self {
        Self {
            note: CompileErrorNote {
                msg: msg.into(),
                range: range.into(),
            },
            extra_notes: Default::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    Symbol(Sym),
    IntLiteral(i64),
    FloatLiteral(f64),
    BoolLiteral(bool),
    StringLiteral(String),
    TypeLiteral(Type),
    Field {
        base: Id,
        field_name: Sym,
        field_index: u16,
    },
    Return(Id),
    Ref(Id),
    Load(Id),
    PtrCast {
        ty: Id,
        expr: Id,
    },
    Let {
        name: Sym,
        ty: Id,
        expr: Option<Id>,
    },
    Set {
        name: Id,
        expr: Id,
        is_store: bool,
    },
    Extern {
        name: Sym,
        params: Vec<Id>,
        return_ty: Id,
    },
    Func {
        name: Sym,
        scope: Id,
        params: Vec<Id>,
        return_ty: Id,
        stmts: Vec<Id>,
    },
    FuncParam {
        name: Sym,
        ty: Id,
        index: u16,
    },
    Struct {
        name: Sym,
        params: Vec<Id>,
    },
    Call {
        name: Id,
        params: Vec<Id>,

        // todo(chad): bitflag?
        is_macro: bool,
        is_indirect: bool,
    },
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    Div(Id, Id),
    LessThan(Id, Id),
    GreaterThan(Id, Id),
    EqualTo(Id, Id),
    And(Id, Id),
    Or(Id, Id),
    Not(Id),
    If(Id, Id, Option<Id>),
    While(Id, Vec<Id>),
}

impl TryInto<Type> for Node {
    type Error = String;

    fn try_into(self) -> Result<Type, Self::Error> {
        match self {
            Node::TypeLiteral(basic_type) => Ok(basic_type.into()),
            _ => Err(String::from("Expected type")),
        }
    }
}

impl Node {
    pub fn as_symbol(&self) -> Option<Sym> {
        match self {
            Node::Symbol(sym) => Some(*sym),
            _ => None,
        }
    }

    pub fn param_field_name(&self) -> Option<Sym> {
        match self {
            Node::FuncParam { name, ty, index } => Some(*name),
            _ => None,
        }
    }
}

impl<'a> Parser<'a> {
    pub fn new(string: &'a str) -> Self {
        let mut lexer = Lexer::new(string);

        // pop first two tokens
        lexer.pop();
        lexer.pop();

        let sym_fn = lexer.string_interner.get_or_intern("fn");
        let sym_struct = lexer.string_interner.get_or_intern("struct");
        let sym_field = lexer.string_interner.get_or_intern("field");
        let sym_let = lexer.string_interner.get_or_intern("let");
        let sym_uninit = lexer.string_interner.get_or_intern("uninit");
        let sym_set = lexer.string_interner.get_or_intern("set");
        let sym_ptr = lexer.string_interner.get_or_intern("ptr");
        let sym_ref = lexer.string_interner.get_or_intern("ref");
        let sym_load = lexer.string_interner.get_or_intern("load");
        let sym_ptrcast = lexer.string_interner.get_or_intern("ptrcast");
        let sym_store = lexer.string_interner.get_or_intern("store");
        let sym_call = lexer.string_interner.get_or_intern("call");
        let sym_return = lexer.string_interner.get_or_intern("return");
        let sym_none = lexer.string_interner.get_or_intern("none");
        let sym_i8 = lexer.string_interner.get_or_intern("i8");
        let sym_i16 = lexer.string_interner.get_or_intern("i16");
        let sym_i32 = lexer.string_interner.get_or_intern("i32");
        let sym_i64 = lexer.string_interner.get_or_intern("i64");
        let sym_f32 = lexer.string_interner.get_or_intern("f32");
        let sym_f64 = lexer.string_interner.get_or_intern("f64");
        let sym_bool = lexer.string_interner.get_or_intern("bool");
        let sym_string = lexer.string_interner.get_or_intern("string");
        let sym_plus = lexer.string_interner.get_or_intern("+");
        let sym_minus = lexer.string_interner.get_or_intern("-");
        let sym_mul = lexer.string_interner.get_or_intern("*");
        let sym_div = lexer.string_interner.get_or_intern("/");
        let sym_lt = lexer.string_interner.get_or_intern("<");
        let sym_gt = lexer.string_interner.get_or_intern(">");
        let sym_eq = lexer.string_interner.get_or_intern("=");
        let sym_and = lexer.string_interner.get_or_intern("and");
        let sym_or = lexer.string_interner.get_or_intern("or");
        let sym_not = lexer.string_interner.get_or_intern("not");
        let sym_if = lexer.string_interner.get_or_intern("if");
        let sym_while = lexer.string_interner.get_or_intern("while");
        let sym_extern = lexer.string_interner.get_or_intern("extern");

        let scopes = vec![Scope {
            parent: None,
            ..Default::default()
        }];

        Parser {
            lexer,
            nodes: Default::default(),
            node_scopes: Default::default(),
            ranges: Default::default(),
            node_is_addressable: Default::default(),
            scopes: scopes,
            top_scope: 0,
            top_level: Default::default(),
            top_level_map: Default::default(),
            sym_fn,
            sym_struct,
            sym_field,
            sym_let,
            sym_uninit,
            sym_set,
            sym_ptr,
            sym_ref,
            sym_load,
            sym_ptrcast,
            sym_store,
            sym_call,
            sym_return,
            sym_none,
            sym_i8,
            sym_i16,
            sym_i32,
            sym_i64,
            sym_f32,
            sym_f64,
            sym_bool,
            sym_string,
            sym_plus,
            sym_minus,
            sym_mul,
            sym_div,
            sym_lt,
            sym_gt,
            sym_eq,
            sym_and,
            sym_or,
            sym_not,
            sym_if,
            sym_while,
            sym_extern,
        }
    }

    pub fn parse(&mut self) -> Result<(), CompileError> {
        while self.lexer.top.tok != Token::EOF {
            let top_level = self.parse_top_level()?;
            self.top_level.push(top_level);
        }

        Ok(())
    }

    pub fn get_top_level(&self, name: impl Into<String>) -> Option<Id> {
        let sym = self.lexer.string_interner.get(name.into()).unwrap();
        self.top_level_map.get(&sym).map(|x| *x)
    }

    pub fn scope_insert(&mut self, sym: Sym, id: Id) {
        self.scopes[self.top_scope].entries.insert(sym, id);
    }

    pub fn local_insert(&mut self, id: Id) {
        self.scopes[self.top_scope].locals.push(id);
    }

    pub fn scope_get_with_scope_id(&self, sym: Sym, scope: Id) -> Option<Id> {
        match self.scopes[scope].entries.get(&sym).map(|x| *x) {
            Some(id) => Some(id),
            None => match self.scopes[scope].parent {
                Some(parent) => self.scope_get_with_scope_id(sym, parent),
                None => None,
            },
        }
    }

    pub fn scope_get(&self, sym: Sym, node: Id) -> Option<Id> {
        let scope_id = self.node_scopes[node];
        self.scope_get_with_scope_id(sym, scope_id)
    }

    pub fn resolve_sym_unchecked(&self, sym: Sym) -> &str {
        self.lexer.resolve_unchecked(sym)
    }

    #[inline]
    pub fn push_node(&mut self, range: Range, node: Node) -> Id {
        self.nodes.push(node);
        self.ranges.push(range);
        self.node_scopes.push(self.top_scope);
        self.node_is_addressable.push(false);

        self.nodes.len() - 1
    }

    #[inline]
    fn parse_symbol(&mut self) -> Result<Id, CompileError> {
        match self.lexer.top.tok {
            Token::Symbol(sym) => {
                self.lexer.pop();
                Ok(self.push_node(self.lexer.top.range, Node::Symbol(sym)))
            }
            _ => Err(CompileError::from_string(
                format!("Expected symbol, got {:?}", self.lexer.top.tok),
                self.lexer.top.range,
            )),
        }
    }

    #[inline]
    fn parse_sym(&mut self) -> Result<Sym, CompileError> {
        let ident = self.parse_symbol()?;
        self.nodes[ident]
            .as_symbol()
            .ok_or(CompileError::from_string(
                String::from("Expected a symbol"),
                self.lexer.top.range,
            ))
    }

    fn parse_type(&mut self) -> Result<Id, CompileError> {
        match self.lexer.top.tok {
            // Token::LParen => {
            //     let start = self.lexer.top.range.start;
            //     self.lexer.pop();

            //     let sym_id = self.parse_symbol()?;
            //     let sym = self.nodes[sym_id].as_symbol().unwrap();
            //     if sym == self.sym_fn {
            //         let mut input_tys = Vec::new();
            //         self.expect(&Token::LParen)?;
            //         while self.lexer.top.tok != Token::RParen {
            //             input_tys.push(self.parse_type()?);
            //         }
            //         self.expect(&Token::RParen)?;

            //         let return_ty = self.parse_type()?;

            //         let range = self.expect_close_paren(start)?;
            //         Ok(self.push_node(
            //             range,
            //             Node::TypeLiteral(Type::Func {
            //                 return_ty,
            //                 input_tys,
            //             }),
            //         ))
            //     } else if sym == self.sym_ptr {
            //         let ptr_ty = self.parse_type()?;
            //         let range = self.expect_close_paren(start)?;
            //         Ok(self.push_node(range, Node::TypeLiteral(Type::Pointer(ptr_ty))))
            //     } else {
            //         Err(CompileError::from_string(
            //             "Failed to parse type",
            //             self.lexer.top.range,
            //         ))
            //     }
            // }
            Token::Asterisk => {
                let start = self.lexer.top.range.start;
                self.lexer.pop(); // `*`

                let pty = self.parse_type()?;
                let end = self.ranges[pty].end;

                Ok(self.push_node(
                    Range::new(start, end),
                    Node::TypeLiteral(Type::Pointer(pty)),
                ))
            }
            Token::Symbol(_) => {
                let ident = self.parse_symbol()?;
                let range = self.ranges[ident];

                let sym = self.nodes[ident].as_symbol().unwrap();
                if sym == self.sym_i8 {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::I8.into())))
                } else if sym == self.sym_none {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::None.into())))
                } else if sym == self.sym_i16 {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::I16.into())))
                } else if sym == self.sym_i32 {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::I32.into())))
                } else if sym == self.sym_i64 {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::I64.into())))
                } else if sym == self.sym_f32 {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::F32.into())))
                } else if sym == self.sym_f64 {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::F64.into())))
                } else if sym == self.sym_bool {
                    Ok(self.push_node(range, Node::TypeLiteral(BasicType::Bool.into())))
                } else if sym == self.sym_string {
                    Ok(self.push_node(range, Node::TypeLiteral(Type::String)))
                } else {
                    Ok(ident)
                }
            }
            _ => Err(CompileError::from_string(
                "Failed to parse type",
                self.lexer.top.range,
            )),
        }
    }

    fn parse_numeric_literal(&mut self) -> Result<Id, CompileError> {
        match self.lexer.top.tok {
            Token::IntegerLiteral(i) => {
                let range = self.lexer.top.range;
                self.lexer.pop();
                Ok(self.push_node(range, Node::IntLiteral(i)))
            }
            Token::FloatLiteral(f) => {
                let range = self.lexer.top.range;
                self.lexer.pop();
                Ok(self.push_node(range, Node::FloatLiteral(f)))
            }
            _ => Err(CompileError::from_string(
                "Expected integer literal",
                self.lexer.top.range,
            )),
        }
    }

    fn parse_if(&mut self, start: Location) -> Result<Id, CompileError> {
        let cond_expr = self.parse_expression()?;
        let true_expr = self.parse_expression()?;

        let false_expr = if self.lexer.top.tok != Token::RParen {
            Some(self.parse_expression()?)
        } else {
            None
        };

        let range = self.expect_close_paren(start)?;
        let if_id = self.push_node(range, Node::If(cond_expr, true_expr, false_expr));

        self.local_insert(if_id);

        Ok(if_id)
    }

    fn parse_while(&mut self, start: Location) -> Result<Id, CompileError> {
        let cond_expr = self.parse_expression()?;

        let mut stmts = Vec::new();
        while let Token::LParen = self.lexer.top.tok {
            stmts.push(self.parse_fn_stmt()?);
        }

        let range = self.expect_close_paren(start)?;

        Ok(self.push_node(range, Node::While(cond_expr, stmts)))
    }

    fn parse_expression(&mut self) -> Result<Id, CompileError> {
        match self.lexer.top.tok.clone() {
            Token::IntegerLiteral(_) | Token::FloatLiteral(_) => self.parse_numeric_literal(),
            Token::StringLiteral(s) => {
                let range = self.lexer.top.range;
                self.lexer.pop();
                Ok(self.push_node(range, Node::StringLiteral(s.clone())))
            }
            // Token::LParen => {
            //     let start = self.lexer.top.range.start;
            //     self.lexer.pop();

            //     let sym = self.parse_sym()?;
            //     if sym == self.sym_plus {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::Add(arg1, arg2)))
            //     } else if sym == self.sym_minus {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::Sub(arg1, arg2)))
            //     } else if sym == self.sym_mul {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::Mul(arg1, arg2)))
            //     } else if sym == self.sym_div {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::Div(arg1, arg2)))
            //     } else if sym == self.sym_lt {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::LessThan(arg1, arg2)))
            //     } else if sym == self.sym_gt {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::GreaterThan(arg1, arg2)))
            //     } else if sym == self.sym_eq {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::EqualTo(arg1, arg2)))
            //     } else if sym == self.sym_and {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::And(arg1, arg2)))
            //     } else if sym == self.sym_or {
            //         let arg1 = self.parse_expression()?;
            //         let arg2 = self.parse_expression()?;

            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::Or(arg1, arg2)))
            //     } else if sym == self.sym_not {
            //         let arg1 = self.parse_expression()?;
            //         let range = self.expect_close_paren(start)?;
            //         Ok(self.push_node(range, Node::Not(arg1)))
            //     } else if sym == self.sym_ref {
            //         let expr = self.parse_expression()?;
            //         let range = self.expect_close_paren(start)?;

            //         Ok(self.push_node(range, Node::Ref(expr)))
            //     } else if sym == self.sym_load {
            //         let expr = self.parse_expression()?;
            //         let range = self.expect_close_paren(start)?;
            //         Ok(self.push_node(range, Node::Load(expr)))
            //     } else if sym == self.sym_ptrcast {
            //         let ty = self.parse_type()?;
            //         let expr = self.parse_expression()?;
            //         let range = self.expect_close_paren(start)?;
            //         Ok(self.push_node(range, Node::PtrCast { ty, expr }))
            //     } else if sym == self.sym_if {
            //         self.parse_if(start)
            //     } else if sym == self.sym_call {
            //         let name = self.parse_symbol()?;
            //         self.parse_call(start, name)
            //     } else if sym == self.sym_field {
            //         let base = self.parse_expression()?;
            //         let field_name = self.parse_sym()?;
            //         let range = self.expect_close_paren(start)?;

            //         let field_id = self.push_node(
            //             range,
            //             Node::Field {
            //                 base,
            //                 field_name,
            //                 field_index: 0,
            //             },
            //         );

            //         self.node_has_slot[field_id] = true;

            //         Ok(field_id)
            //     } else {
            //         Err(CompileError::from_string(
            //             "Unexpected",
            //             self.lexer.top.range,
            //         ))
            //     }
            // }
            Token::Ampersand => {
                let start = self.lexer.top.range.start;
                self.lexer.pop(); // `&`

                let expr = self.parse_expression()?;
                self.node_is_addressable[expr] = true;
                let end = self.ranges[expr].end;

                Ok(self.push_node(Range::new(start, end), Node::Ref(expr)))
            }
            Token::Caret => {
                let start = self.lexer.top.range.start;
                self.lexer.pop(); // `^`

                let expr = self.parse_expression()?;
                self.node_is_addressable[expr] = true;
                let end = self.ranges[expr].end;

                Ok(self.push_node(Range::new(start, end), Node::Load(expr)))
            }
            Token::True => {
                let range = self.lexer.top.range;
                self.lexer.pop(); // `true`
                Ok(self.push_node(range, Node::BoolLiteral(true)))
            }
            Token::False => {
                let range = self.lexer.top.range;
                self.lexer.pop(); // `false`
                Ok(self.push_node(range, Node::BoolLiteral(false)))
            }
            _ => {
                let start = self.lexer.top.range.start;

                let mut value = self.parse_lvalue()?;

                // dot / function call?
                while self.lexer.top.tok == Token::Dot || self.lexer.top.tok == Token::LParen {
                    if self.lexer.top.tok == Token::Dot {
                        self.lexer.pop(); // `.`
                        let field_name = self.parse_symbol()?;

                        let start = self.ranges[value].start;
                        let end = self.ranges[field_name].end;

                        let field_name = self.nodes[field_name].as_symbol().unwrap();

                        value = self.push_node(
                            Range::new(start, end),
                            Node::Field {
                                base: value,
                                field_name,
                                field_index: 0,
                            },
                        );

                        // all field accesses represent addressable memory
                        self.node_is_addressable[value] = true;
                    } else {
                        self.expect(&Token::LParen)?;
                        let params = self.parse_value_params()?;
                        let range = self.expect_range(start, Token::RParen)?;

                        value = self.push_node(
                            range,
                            Node::Call {
                                name: value,
                                params,
                                is_indirect: false,
                                is_macro: false,
                            },
                        );
                    }
                }

                Ok(value)
            }
        }
    }

    fn parse_lvalue(&mut self) -> Result<Id, CompileError> {
        if self.lexer.top.tok == Token::LParen {
            let start = self.lexer.top.range.start;
            self.lexer.pop(); // `(`
            let expr = self.parse_expression()?;
            self.expect(&Token::RParen)?;

            Ok(expr)
        } else if self.lexer.top.tok == Token::Caret {
            let start = self.lexer.top.range.start;
            self.lexer.pop(); // `^`

            let expr = self.parse_expression()?;
            let end = self.ranges[expr].end;

            Ok(self.push_node(Range::new(start, end), Node::Load(expr)))
        } else if let Token::Symbol(_) = self.lexer.top.tok {
            Ok(self.parse_symbol()?)
        } else {
            Err(CompileError::from_string(
                "Failed to parse lvalue",
                self.lexer.top.range,
            ))
        }
    }

    fn expect_close_paren(&mut self, start: Location) -> Result<Range, CompileError> {
        let range = Range::new(start, self.lexer.top.range.end);

        match self.lexer.top.tok {
            Token::RParen => {
                self.lexer.pop();
                Ok(range)
            }
            _ => Err(CompileError::from_string("Expected ')'", range)),
        }
    }

    fn expect_range(&mut self, start: Location, token: Token) -> Result<Range, CompileError> {
        let range = Range::new(start, self.lexer.top.range.end);

        if self.lexer.top.tok == token {
            self.lexer.pop();
            Ok(range)
        } else {
            Err(CompileError::from_string(
                format!("Expected '{:?}', found {:?}", token, self.lexer.top.tok),
                range,
            ))
        }
    }

    fn parse_fn_stmt(&mut self) -> Result<Id, CompileError> {
        let start = self.lexer.top.range.start;

        match self.lexer.top.tok {
            // todo(chad): turn `return`, `let`, etc. into their own tokens rather than parsing as symbols
            Token::Symbol(sym) if sym == self.sym_return => {
                self.lexer.pop(); // `return`
                let expr = self.parse_expression()?;
                let range = self.expect_range(start, Token::Semicolon)?;

                Ok(self.push_node(range, Node::Return(expr)))
            }
            Token::Symbol(sym) if sym == self.sym_let => {
                self.lexer.pop(); // `let`
                let name = self.parse_sym()?;
                self.expect(&Token::Colon)?;
                let ty = self.parse_type()?;

                self.expect(&Token::Eq)?;

                let expr = match self.lexer.top.tok {
                    Token::Uninit => {
                        self.lexer.pop(); // pop 'uninit' token
                        None
                    }
                    _ => Some(self.parse_expression()?),
                };

                let range = self.expect_range(start, Token::Semicolon)?;
                let let_id = self.push_node(range, Node::Let { name, ty, expr });
                self.node_is_addressable[let_id] = true;
                self.scope_insert(name, let_id);
                self.local_insert(let_id);

                Ok(let_id)
            }
            _ => {
                let lvalue = self.parse_expression()?;

                match self.lexer.top.tok {
                    // Assignment?
                    Token::Eq => {
                        // parsing something like "foo = expr;";
                        self.expect(&Token::Eq)?;
                        let expr = self.parse_expression()?;
                        let range = self.expect_range(start, Token::Semicolon)?;

                        Ok(self.push_node(
                            range,
                            Node::Set {
                                name: lvalue,
                                expr,
                                is_store: false,
                            },
                        ))
                    }
                    _ => Ok(lvalue),
                }
            }
        }
    }

    fn expect(&mut self, tok: &Token) -> Result<(), CompileError> {
        match &self.lexer.top.tok {
            t if t == tok => {
                self.lexer.pop();
                Ok(())
            }
            _ => Err(CompileError::from_string(
                format!("Expected {:?}, got {:?}", tok, self.lexer.top.tok),
                self.lexer.top.range,
            )),
        }
    }

    fn parse_params(&mut self) -> Result<Vec<Id>, CompileError> {
        let mut params = Vec::new();

        let mut index = 0;
        while self.lexer.top.tok != Token::RParen && self.lexer.top.tok != Token::RCurly {
            let input_start = self.lexer.top.range.start;

            let name = self.parse_symbol()?;
            let name_sym = self.nodes[name].as_symbol().unwrap();
            self.expect(&Token::Colon)?;
            let ty = self.parse_type()?;

            let range = Range::new(input_start, self.ranges[ty].end);

            // put the param in scope
            let param = self.push_node(
                range,
                Node::FuncParam {
                    name: name_sym,
                    ty,
                    index,
                },
            );
            self.scope_insert(name_sym, param);
            params.push(param);

            if self.lexer.top.tok == Token::Comma {
                self.lexer.pop(); // `,`
            }

            index += 1;
        }

        Ok(params)
    }

    fn parse_value_params(&mut self) -> Result<Vec<Id>, CompileError> {
        let mut params = Vec::new();

        while self.lexer.top.tok != Token::RParen && self.lexer.top.tok != Token::RCurly {
            params.push(self.parse_expression()?);

            if self.lexer.top.tok == Token::Comma {
                self.lexer.pop(); // `,`
            }
        }

        Ok(params)
    }

    fn parse_fn(&mut self, start: Location) -> Result<Id, CompileError> {
        self.expect(&Token::Symbol(self.sym_fn))?;

        let name = self.parse_sym()?;

        // open a new scope
        self.scopes.push(Scope::new(self.top_scope));

        let old_top_scope = self.top_scope;
        self.top_scope = self.scopes.len() - 1;

        let params_start = self.lexer.top.range.start;
        self.expect(&Token::LParen)?;
        let params = self.parse_params()?;
        self.expect_close_paren(params_start)?;

        let return_ty = self.parse_type()?;

        self.expect(&Token::LCurly)?;

        let mut stmts = Vec::new();
        while self.lexer.top.tok != Token::RCurly {
            stmts.push(self.parse_fn_stmt()?);
        }

        let range = self.expect_range(start, Token::RCurly)?;

        let func = self.push_node(
            range,
            Node::Func {
                name,
                scope: self.top_scope,
                params,
                return_ty,
                stmts,
            },
        );

        // pop the top scope
        // TODO: should probably do this in a 'defer', so that the scope is also popped in the event of an error
        self.top_scope = old_top_scope;

        self.scope_insert(name, func);
        self.top_level_map.insert(name, func);

        Ok(func)
    }

    fn parse_top_level(&mut self) -> Result<Id, CompileError> {
        let start = self.lexer.top.range.start;

        match self.lexer.top.tok {
            Token::Symbol(sym) if sym == self.sym_struct => {
                self.lexer.pop();
                let name = self.parse_sym()?;

                self.expect(&Token::LCurly)?;
                let params = self.parse_params()?;
                let range = self.expect_range(start, Token::RCurly)?;

                let id = self.push_node(range, Node::Struct { name, params });
                self.scope_insert(name, id);

                Ok(id)
            }
            Token::Symbol(sym) if sym == self.sym_fn => self.parse_fn(start),
            // Token::LParen => {
            //     let start = self.lexer.top.range.start;

            //     self.lexer.pop();

            //     let sym = self.parse_sym()?;
            //     if sym == self.sym_fn {
            //         self.parse_func(start)
            //     } else if sym == self.sym_extern {
            //         let name = self.parse_sym()?;

            //         let mut params = Vec::new();
            //         let params_start = self.lexer.top.range.start;
            //         self.expect(&Token::LParen)?;
            //         let mut index = 0;
            //         while self.lexer.top.tok != Token::RParen {
            //             let input_start = self.lexer.top.range.start;
            //             self.expect(&Token::LParen)?;

            //             let name = self.parse_symbol()?;
            //             let name_sym = self.nodes[name].as_symbol().unwrap();
            //             let ty = self.parse_type()?;

            //             let range = self.expect_close_paren(input_start)?;

            //             let param = self.push_node(
            //                 range,
            //                 Node::FuncParam {
            //                     name: name_sym,
            //                     ty,
            //                     index,
            //                 },
            //             );
            //             params.push(param);

            //             index += 1;
            //         }
            //         self.expect_close_paren(params_start)?;

            //         let return_ty = self.parse_type()?;
            //         let range = self.expect_close_paren(start)?;

            //         let node_id = self.push_node(
            //             range,
            //             Node::Extern {
            //                 name,
            //                 params,
            //                 return_ty,
            //             },
            //         );

            //         self.scope_insert(name, node_id);

            //         Ok(node_id)
            //     } else {
            //         Err(CompileError::from_string(
            //             format!("Unrecognized Symbol {}", self.resolve_sym_unchecked(sym)),
            //             self.lexer.top.range,
            //         ))
            //     }
            // }
            _ => Err(CompileError::from_string(
                "Unexpected top level",
                self.lexer.top.range,
            )),
        }
    }

    pub fn debug_sym(&self, sym: Sym) -> String {
        String::from(self.lexer.string_interner.resolve(sym).unwrap())
    }

    pub fn debug(&self, id: Id) -> String {
        match &self.nodes[id] {
            Node::Symbol(sym) => self.debug_sym(*sym),
            Node::IntLiteral(i) => format!("{}", i),
            Node::FloatLiteral(f) => format!("{}", f),
            Node::BoolLiteral(b) => format!("{}", b),
            Node::StringLiteral(s) => format!("\"{}\"", s),
            Node::TypeLiteral(ty) => match ty {
                Type::Pointer(p) => format!("(ptr {})", self.debug(*p)),
                _ => format!("{:?}", ty),
            },
            Node::Return(id) => format!("(return {})", self.debug(*id)),
            Node::Let { name, ty, expr } => format!(
                "(let {} {} {})",
                self.debug_sym(*name),
                self.debug(*ty),
                if expr.is_some() {
                    self.debug(expr.unwrap())
                } else {
                    String::from("uninit")
                }
            ),
            Node::Set { name, expr, .. } => {
                format!("(set {} {})", self.debug_sym(*name), self.debug(*expr))
            }
            Node::Ref(id) => format!("(ref {})", self.debug(*id)),
            Node::Field {
                base, field_name, ..
            } => format!(
                "(field {} {})",
                self.debug(*base),
                self.debug_sym(*field_name)
            ),
            Node::Load(id) => format!("(load {})", self.debug(*id)),
            Node::PtrCast { ty, expr } => {
                format!("(ptrcast {} {})", self.debug(*ty), self.debug(*expr))
            }
            Node::Func {
                name,
                scope: _,
                params,
                return_ty,
                stmts,
            } => {
                let mut d = format!(
                    "(fn {} ({}) {} ",
                    self.debug_sym(*name),
                    params
                        .iter()
                        .map(|p| self.debug(*p))
                        .collect::<Vec<_>>()
                        .join(","),
                    self.debug(*return_ty)
                );
                for bb in stmts {
                    d += "\n  ";
                    d += &self.debug(*bb)
                }
                d += ")";
                d
            }
            Node::Struct { name, params } => format!(
                "(struct {} ({}))",
                self.debug_sym(*name),
                params
                    .iter()
                    .map(|p| self.debug(*p))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Node::Extern {
                name,
                params,
                return_ty,
            } => {
                let mut d = format!(
                    "(extern {} ({}) {} ",
                    self.debug_sym(*name),
                    params
                        .iter()
                        .map(|p| self.debug(*p))
                        .collect::<Vec<_>>()
                        .join(","),
                    self.debug(*return_ty)
                );
                d += ")";
                d
            }
            Node::FuncParam { name, ty, index: _ } => {
                format!("({} {})", self.debug_sym(*name), self.debug(*ty))
            }
            Node::Call {
                name,
                params,
                is_macro,
                is_indirect,
            } => format!(
                "({} {} {})",
                if *is_macro {
                    "callm"
                } else if *is_indirect {
                    "calli"
                } else {
                    "call"
                },
                self.debug(*name),
                params
                    .iter()
                    .map(|a| self.debug(*a))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Node::Add(arg1, arg2) => format!("(+ {} {})", self.debug(*arg1), self.debug(*arg2)),
            Node::Sub(arg1, arg2) => format!("(- {} {})", self.debug(*arg1), self.debug(*arg2)),
            Node::Mul(arg1, arg2) => format!("(* {} {})", self.debug(*arg1), self.debug(*arg2)),
            Node::Div(arg1, arg2) => format!("(/ {} {})", self.debug(*arg1), self.debug(*arg2)),
            Node::LessThan(arg1, arg2) => {
                format!("(< {} {})", self.debug(*arg1), self.debug(*arg2))
            }
            Node::GreaterThan(arg1, arg2) => {
                format!("(> {} {})", self.debug(*arg1), self.debug(*arg2))
            }
            Node::EqualTo(arg1, arg2) => format!("(= {} {})", self.debug(*arg1), self.debug(*arg2)),
            Node::And(arg1, arg2) => format!("(and {} {})", self.debug(*arg1), self.debug(*arg2)),
            Node::Or(arg1, arg2) => format!("(or {} {})", self.debug(*arg1), self.debug(*arg2)),
            Node::Not(arg1) => format!("(not {})", self.debug(*arg1)),
            Node::If(c, i, e) => format!(
                "(if {} {} {})",
                self.debug(*c),
                self.debug(*i),
                e.map(|m| self.debug(m)).unwrap_or_default()
            ),
            Node::While(c, e) => format!(
                "(while {} {})",
                self.debug(*c),
                e.iter()
                    .map(|s| self.debug(*s))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        }
    }
}
