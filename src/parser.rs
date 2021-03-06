use std::collections::BTreeMap;
use std::convert::TryInto;

use string_interner::StringInterner;

type Sym = usize;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdVec(pub Id);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenVec(pub Id);

#[derive(Clone, Copy, PartialEq)]
pub struct Location {
    pub line: usize,
    pub col: usize,
    pub char_offset: usize,
}

impl std::fmt::Debug for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

impl Default for Location {
    fn default() -> Self {
        Self {
            line: 1,
            col: 0,
            char_offset: 0,
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Range {
    pub start: Location,
    pub end: Location,
}

impl Range {
    pub fn new(start: Location, end: Location) -> Self {
        Self { start, end }
    }

    pub fn spanning(start: Range, end: Range) -> Self {
        Self {
            start: start.start,
            end: end.end,
        }
    }
}

impl std::fmt::Debug for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} - {:?}", self.start, self.end)
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
    IntLiteral,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
    Unassigned,
    Basic(BasicType),
    Pointer(Id),
    String,
    Func {
        return_ty: Id,
        input_tys: IdVec,
        copied_from: Option<Id>,
    },
    Struct {
        params: IdVec,
        copied_from: Option<Id>,
    },
    StructLiteral(IdVec),
    Enum {
        params: IdVec,
        copied_from: Option<Id>,
    },
    Array(Id),
    Type,
}

impl Type {
    pub fn as_struct_params(&self) -> Option<IdVec> {
        match self {
            Type::Struct { params, .. } => Some(*params),
            Type::Enum { params, .. } => Some(*params),
            _ => None,
        }
    }

    // pub fn is_basic(&self) -> bool {
    //     match self {
    //         Type::Basic(_) => true,
    //         _ => false,
    //     }
    // }

    // pub fn is_coercble(&self) -> bool {
    //     if self.is_concrete() {
    //         return true;
    //     }

    //     match self {
    //         Type::Basic(BasicType::IntLiteral) => true,
    //         _ => false,
    //     }
    // }
}

impl Into<Type> for BasicType {
    fn into(self) -> Type {
        Type::Basic(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericSpecification {
    None,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token {
    LParen,
    RParen,
    LCurly,
    RCurly,
    LSquare,
    RSquare,
    DoubleLAngle,
    DoubleRAngle,
    LAngle,
    RAngle,
    Semicolon,
    Colon,
    Bang,
    Dot,
    Asterisk,
    EqEq,
    Neq,
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Comma,
    Import,
    Struct,
    Enum,
    Fn,
    Macro,
    Extern,
    TypeOf,
    Cast,
    Let,
    If,
    Else,
    While,
    Return,
    True,
    False,
    Bool,
    String,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    None,
    Ampersand,
    Type,
    Caret,
    Uninit,
    Symbol(Sym),
    IntegerLiteral(i64, NumericSpecification), // TODO: handle negative literals
    FloatLiteral(f64),                         // TODO: handle negative literals
    StringLiteral(Sym),
    Eof,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lexeme {
    pub tok: Token,
    pub range: Range,
}

impl Lexeme {
    pub fn new(tok: Token, range: Range) -> Self {
        Self { tok, range }
    }
}

#[derive(Debug)]
struct PushedScope(Id, bool);

#[derive(Clone)]
pub struct Lexer<'a> {
    // TODO(chad): PLAN
    // - Move string_interner out of lexer
    // - Make source_info a borrowed member
    // - Then everything else should be ephemeral
    // - We can then just spin-up lexers when we need them and throw them away when we're done
    // - Then, make `lexer` a private field in the `Parser` struct (or remove it entirely and pass functionally to all the parsing methods)
    pub string_interner: Box<StringInterner<Sym>>,

    pub source_info: SourceInfo<'a>,
    pub source: &'a str,
    pub loc: Location,

    pub macro_tokens: Option<Vec<Lexeme>>,

    pub top: Lexeme,
    pub second: Lexeme,
}

#[derive(Clone)]
pub struct SourceInfo<'a> {
    pub source_name: String,
    pub original_source: &'a str,
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
        || c == '<'
        || c == '>'
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
    pub fn new(source_name: String, source: &'a str) -> Self {
        let string_interner = Box::new(StringInterner::new());

        Self {
            string_interner,
            source_info: SourceInfo {
                source_name,
                original_source: source,
            },
            source: source,
            loc: Default::default(),
            top: Lexeme::new(Token::Eof, Default::default()),
            second: Lexeme::new(Token::Eof, Default::default()),
            macro_tokens: None,
        }
    }

    pub fn update_source_for_copy(&mut self, source: &'a str, loc: Location) {
        self.source = source;
        self.loc = loc;
        self.top = Lexeme::new(Token::Eof, Default::default());
        self.second = Lexeme::new(Token::Eof, Default::default());

        self.pop();
        self.pop();
    }

    #[inline]
    fn eat(&mut self, chars: usize) {
        self.source = &self.source[chars..];
        self.loc.col += chars;
        self.loc.char_offset += chars;
    }

    #[inline]
    fn newline(&mut self) {
        self.source = &self.source[1..];
        self.loc.line += 1;
        self.loc.col = 1;
        self.loc.char_offset += 1;
    }

    #[inline]
    fn eat_rest_of_line(&mut self) {
        let chars = self
            .source
            .chars()
            .position(|c| c == '\n')
            .unwrap_or_else(|| self.source.len());
        self.eat(chars);

        if !self.source.is_empty() {
            self.newline();
        }
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

    pub fn resolve_unchecked(&self, sym: Sym) -> &str {
        self.string_interner.resolve(sym).unwrap()
    }

    fn prefix(&mut self, pat: &str, tok: Token) -> bool {
        if self.source.len() >= pat.len() && self.source.starts_with(pat) {
            let start = self.loc;
            self.eat(pat.len());
            self.second = Lexeme::new(tok, Range::new(start, self.loc));
            true
        } else {
            false
        }
    }

    fn prefix_keyword(&mut self, pat: &str, tok: Token) -> bool {
        if self.source.len() > pat.len()
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
        if let Some(macro_tokens) = self.macro_tokens.as_mut() {
            let macro_tokens: &mut Vec<Lexeme> = macro_tokens;

            self.top = self.second;
            self.second = macro_tokens
                .first()
                .cloned()
                .unwrap_or_else(|| Lexeme::new(Token::Eof, Range::default()));

            if !macro_tokens.is_empty() {
                macro_tokens.remove(0);
            }

            return;
        }

        self.eat_spaces();

        let start = self.loc;

        self.top = self.second;

        // check_keywords(
        //     "---" => Token::Uninit,
        //     "fn" => Token::Fn,
        //     "struct" => Token::Struct,
        //     "true" => Token::True,
        //     "false" => Token::False,
        // )

        // check_tokens! {
        //     "(" => Token::LParen,
        //     ")" => Token::RParen,
        //     "{" => Token::LCurly,
        //     "}" => Token::RCurly,
        //     "," => Token::Comma,
        //     "&" => Token::Ampersand,
        //     "." => Token::Dot,
        //     "^" => Token::Caret,
        //     ";" => Token::Semicolon,
        //     "*" => Token::Asterisk,
        //     ":" => Token::Colon,
        //     "=" => Token::Eq,
        // }

        if self.prefix_keyword("---", Token::Uninit) {
            return;
        }
        if self.prefix_keyword("fn", Token::Fn) {
            return;
        }
        if self.prefix_keyword("macro", Token::Macro) {
            return;
        }
        if self.prefix_keyword("extern", Token::Extern) {
            return;
        }
        if self.prefix_keyword("let", Token::Let) {
            return;
        }
        if self.prefix_keyword("if", Token::If) {
            return;
        }
        if self.prefix_keyword("else", Token::Else) {
            return;
        }
        if self.prefix_keyword("while", Token::While) {
            return;
        }
        if self.prefix_keyword("return", Token::Return) {
            return;
        }
        if self.prefix_keyword("#cast", Token::Cast) {
            return;
        }
        if self.prefix_keyword("#type_of", Token::TypeOf) {
            return;
        }
        if self.prefix_keyword("none", Token::None) {
            return;
        }
        if self.prefix_keyword("bool", Token::Bool) {
            return;
        }
        if self.prefix_keyword("string", Token::String) {
            return;
        }
        if self.prefix_keyword("i8", Token::I8) {
            return;
        }
        if self.prefix_keyword("i16", Token::I16) {
            return;
        }
        if self.prefix_keyword("i32", Token::I32) {
            return;
        }
        if self.prefix_keyword("i64", Token::I64) {
            return;
        }
        if self.prefix_keyword("f32", Token::F32) {
            return;
        }
        if self.prefix_keyword("f64", Token::F64) {
            return;
        }
        if self.prefix_keyword("import", Token::Import) {
            return;
        }
        if self.prefix_keyword("struct", Token::Struct) {
            return;
        }
        if self.prefix_keyword("Type", Token::Type) {
            return;
        }
        if self.prefix_keyword("enum", Token::Enum) {
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
        if self.prefix("[", Token::LSquare) {
            return;
        }
        if self.prefix("]", Token::RSquare) {
            return;
        }
        if self.prefix("<<", Token::DoubleLAngle) {
            return;
        }
        if self.prefix(">>", Token::DoubleRAngle) {
            return;
        }
        if self.prefix("<", Token::LAngle) {
            return;
        }
        if self.prefix(">", Token::RAngle) {
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
        if self.prefix("!=", Token::Neq) {
            return;
        }
        if self.prefix("!", Token::Bang) {
            return;
        }
        if self.prefix("==", Token::EqEq) {
            return;
        }
        if self.prefix("=", Token::Eq) {
            return;
        }
        if self.prefix("+", Token::Add) {
            return;
        }
        if self.prefix("-", Token::Sub) {
            return;
        }
        if self.prefix("/", Token::Div) {
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
                let s = s.replace("\\n", "\n");
                let s = self.string_interner.get_or_intern(s);

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
                    let mut spec = NumericSpecification::None;

                    if self.source.starts_with("i8") {
                        spec = NumericSpecification::I8;
                        self.eat(2);
                    } else if self.source.starts_with("i16") {
                        spec = NumericSpecification::I16;
                        self.eat(3);
                    } else if self.source.starts_with("i32") {
                        spec = NumericSpecification::I32;
                        self.eat(3);
                    } else if self.source.starts_with("i64") {
                        spec = NumericSpecification::I64;
                        self.eat(3);
                    }

                    let end = self.loc;
                    Lexeme::new(Token::IntegerLiteral(digit, spec), Range::new(start, end))
                }
            }
            Some(_) => {
                let index = match self.source.chars().position(is_special) {
                    Some(index) => index,
                    None => self.source.len(),
                };

                if index == 0 {
                    Lexeme::new(Token::Eof, Default::default())
                } else {
                    let sym = self.string_interner.get_or_intern(&self.source[..index]);
                    self.eat(index);

                    let end = self.loc;

                    Lexeme::new(Token::Symbol(sym), Range::new(start, end))
                }
            }
            None => Lexeme::new(Token::Eof, Default::default()),
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

    // We need a way of passing a node's id and getting the index of the source_info/lexer we are after
    // (since there will be so much duplication)
    pub lexers: Vec<Lexer<'a>>,

    pub node_is_addressable: Vec<bool>,

    pub node_scopes: Vec<Id>,

    pub scopes: Vec<Scope>,
    pub top_scope: Id,
    pub unquote_scope: Id,

    pub function_scopes: Vec<Id>,
    pub macro_calls: Vec<Id>,
    pub function_by_macro_call: BTreeMap<Id, Id>,
    pub macro_calls_by_function: BTreeMap<Id, Vec<Id>>,

    pub returns: Vec<Id>,
    pub string_literals: Vec<Id>,

    pub is_in_insert: bool,
    pub current_unquote: IdVec,
    pub parsed_unquotes: Vec<Id>,

    // todo(chad): are these worth flattening?
    pub id_vecs: Vec<Vec<Id>>,
    pub token_vecs: Vec<Vec<Lexeme>>,

    pub copying: bool,

    // todo(chad): represent list of locals per function
    pub top_level: Vec<Id>,
    pub macros: Vec<Id>,
    pub externs: Vec<Id>,
    pub calls: Vec<Id>,
    pub top_level_map: BTreeMap<Sym, Id>,

    pub ty_decl: Option<Id>,
}

#[derive(Debug, Clone)]
pub struct CompileErrorNote {
    pub msg: String,
    pub range: Range,
}

#[derive(Debug, Clone)]
pub struct CompileError {
    pub note: CompileErrorNote,
}

impl CompileError {
    pub fn from_string(msg: impl Into<String>, range: impl Into<Range>) -> Self {
        Self {
            note: CompileErrorNote {
                msg: msg.into(),
                range: range.into(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Node {
    Symbol(Sym),
    IntLiteral(i64, NumericSpecification),
    FloatLiteral(f64),
    BoolLiteral(bool),
    StringLiteral {
        sym: Sym,
        bytes: usize,
    },
    TypeLiteral(Type),
    If {
        cond: Id,
        true_stmts: IdVec,
        false_stmts: IdVec,
    },
    While {
        cond: Id,
        stmts: IdVec,
    },
    Field {
        base: Id,
        field_name: Sym,
        field_index: u16,
        is_assignment: bool,
    },
    Return(Id),
    Ref(Id),
    Load(Id),
    Let {
        name: Sym,
        ty: Option<Id>,
        expr: Option<Id>,
    },
    Set {
        name: Id,
        expr: Id,
        is_store: bool,
    },
    Func {
        name: Sym,
        scope: Id,
        ct_params: Option<IdVec>,
        params: IdVec,
        return_ty: Id,
        stmts: IdVec,
        returns: IdVec,
        is_macro: bool,
        copied_from: Option<Id>,
    },
    Import {
        name: Sym,
    },
    Extern {
        name: Sym,
        params: IdVec,
        return_ty: Id,
    },
    DeclParam {
        name: Sym,
        ty: Id,
        index: u16,
        is_ct: bool,
        ct_link: Option<Id>,
    },
    Struct {
        name: Sym,
        ct_params: Option<IdVec>,
        params: IdVec,
    },
    Enum {
        name: Sym,
        ct_params: Option<IdVec>,
        params: IdVec,
    },
    ValueParam {
        name: Option<Sym>,
        value: Id,
        index: u16,
        is_ct: bool,
    },
    StructLiteral {
        name: Option<Sym>,
        params: IdVec,
    },
    ArrayLiteral(IdVec),
    Call {
        name: Id,
        ct_params: Option<IdVec>,
        params: IdVec,
    },
    MacroCall {
        name: Id,
        input: TokenVec,
        expanded: IdVec,
    },
    ArrayAccess {
        arr: Id,
        index: Id,
    },
    TypeOf(Id),
    Cast(Id),
    PushToken(Id, Id),
    EqEq(Id, Id),
    Neq(Id, Id),
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    Div(Id, Id),
    LessThan(Id, Id),
    GreaterThan(Id, Id),
    // EqualTo(Id, Id),
    // And(Id, Id),
    // Or(Id, Id),
    // Not(Id),
    // While {
    //     cond: Id,
    //     stmts: IdVec,
    // },
}

impl TryInto<Type> for Node {
    type Error = String;

    fn try_into(self) -> Result<Type, Self::Error> {
        match self {
            Node::TypeLiteral(basic_type) => Ok(basic_type),
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

    pub fn as_string_literal(&self) -> Option<Sym> {
        match self {
            Node::StringLiteral { sym, bytes: _ } => Some(*sym),
            _ => None,
        }
    }
}

impl<'a> Parser<'a> {
    pub fn new(source_name: String, string: &'a str) -> Self {
        let mut lexer = Lexer::new(source_name.clone(), string);

        // pop first two tokens
        lexer.pop();
        lexer.pop();

        let scopes = vec![Scope {
            parent: None,
            ..Default::default()
        }];

        Parser {
            lexers: vec![lexer],
            nodes: Default::default(),
            node_scopes: Default::default(),
            ranges: Default::default(),
            node_is_addressable: Default::default(),
            scopes,
            top_scope: 0,
            function_scopes: vec![0],
            unquote_scope: 0,
            copying: false,
            top_level: Default::default(),
            macros: Default::default(),
            externs: Default::default(),
            calls: Default::default(),
            top_level_map: Default::default(),
            id_vecs: Default::default(),
            token_vecs: Default::default(),
            returns: Default::default(),
            macro_calls: Default::default(),
            function_by_macro_call: Default::default(),
            macro_calls_by_function: Default::default(),
            string_literals: Default::default(),
            current_unquote: IdVec(0),
            parsed_unquotes: Default::default(),
            is_in_insert: false,
            ty_decl: None,
        }
    }

    pub fn parse(&mut self) -> Result<(), CompileError> {
        while self.lexers[0].top.tok != Token::Eof {
            let top_level = self.parse_top_level()?;
            self.top_level.push(top_level);
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_top_level(&self, name: impl Into<String>) -> Option<Id> {
        let sym = self.lexers[0].string_interner.get(name.into()).unwrap();
        self.top_level_map.get(&sym).copied()
    }

    pub fn scope_insert(&mut self, sym: Sym, id: Id) {
        self.scopes[self.top_scope].entries.insert(sym, id);
    }

    pub fn local_insert(&mut self, id: Id) {
        self.scopes[self.top_scope].locals.push(id);
    }

    pub fn scope_get_with_scope_id(&self, sym: Sym, scope: Id) -> Option<Id> {
        match self.scopes[scope].entries.get(&sym).copied() {
            Some(id) => Some(id),
            None => {
                if scope != 0 {
                    match self.scopes[scope].parent {
                        Some(parent) => self.scope_get_with_scope_id(sym, parent),
                        None => None,
                    }
                } else {
                    None
                }
            }
        }
    }

    pub fn scope_get(&self, sym: Sym, node: Id) -> Option<Id> {
        let scope_id = self.node_scopes[node];
        self.scope_get_with_scope_id(sym, scope_id)
    }

    pub fn resolve_sym_unchecked(&self, sym: Sym) -> &str {
        self.lexers[0].resolve_unchecked(sym)
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
        let range = self.lexers[0].top.range;
        match self.lexers[0].top.tok {
            Token::Symbol(sym) => {
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::Symbol(sym)))
            }
            _ => Err(CompileError::from_string(
                format!("Expected symbol, got {:?}", self.lexers[0].top.tok),
                range,
            )),
        }
    }

    #[inline]
    fn parse_sym(&mut self) -> Result<Sym, CompileError> {
        let ident = self.parse_symbol()?;
        self.nodes[ident].as_symbol().ok_or_else(|| {
            CompileError::from_string(String::from("Expected a symbol"), self.lexers[0].top.range)
        })
    }

    fn parse_type(&mut self) -> Result<Id, CompileError> {
        match self.lexers[0].top.tok {
            Token::Fn => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `fn`

                self.expect(&Token::LParen)?;
                let input_tys = self.parse_decl_params(false)?;
                self.expect(&Token::RParen)?;

                let return_ty = self.parse_type()?;
                let end = self.ranges[return_ty].end;

                Ok(self.push_node(
                    Range::new(start, end),
                    Node::TypeLiteral(Type::Func {
                        return_ty,
                        input_tys,
                        copied_from: None,
                    }),
                ))
            }
            Token::Struct => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `struct`

                self.expect(&Token::LCurly)?;
                let params = self.parse_decl_params(false)?;
                let range = self.expect_range(start, Token::RCurly)?;

                Ok(self.push_node(
                    range,
                    Node::TypeLiteral(Type::Struct {
                        params,
                        copied_from: None,
                    }),
                ))
            }
            Token::Enum => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `enum`

                self.expect(&Token::LCurly)?;
                let params = self.parse_decl_params(false)?;
                let range = self.expect_range(start, Token::RCurly)?;

                Ok(self.push_node(
                    range,
                    Node::TypeLiteral(Type::Enum {
                        params,
                        copied_from: None,
                    }),
                ))
            }
            Token::LSquare => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `[`
                self.expect(&Token::RSquare)?;

                let array_ty = self.parse_type()?;
                let range = Range::new(start, self.ranges[array_ty].end);

                Ok(self.push_node(range, Node::TypeLiteral(Type::Array(array_ty))))
            }
            Token::Asterisk => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `*`

                let pty = self.parse_type()?;
                let end = self.ranges[pty].end;

                Ok(self.push_node(
                    Range::new(start, end),
                    Node::TypeLiteral(Type::Pointer(pty)),
                ))
            }
            Token::Type => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop(); // `Type`

                Ok(self.push_node(range, Node::TypeLiteral(Type::Type)))
            }
            Token::I8 => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::I8.into())))
            }
            Token::I16 => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::I16.into())))
            }
            Token::I32 => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::I32.into())))
            }
            Token::I64 => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::I64.into())))
            }
            Token::None => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::None.into())))
            }
            Token::F32 => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::F32.into())))
            }
            Token::F64 => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::F64.into())))
            }
            Token::Bool => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(BasicType::Bool.into())))
            }
            Token::String => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::TypeLiteral(Type::String)))
            }
            Token::Symbol(_) => {
                let ident = self.parse_symbol()?;
                Ok(ident)
            }
            _ => Err(CompileError::from_string(
                "Failed to parse type",
                self.lexers[0].top.range,
            )),
        }
    }

    fn parse_numeric_literal(&mut self) -> Result<Id, CompileError> {
        match self.lexers[0].top.tok {
            Token::IntegerLiteral(i, s) => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::IntLiteral(i, s)))
            }
            Token::FloatLiteral(f) => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop();
                Ok(self.push_node(range, Node::FloatLiteral(f)))
            }
            _ => Err(CompileError::from_string(
                "Expected integer literal",
                self.lexers[0].top.range,
            )),
        }
    }

    fn parse_if(&mut self) -> Result<Id, CompileError> {
        let start = self.lexers[0].top.range.start;

        self.expect(&Token::If)?;

        let cond_expr = match (&self.lexers[0].top.tok, &self.lexers[0].second.tok) {
            (Token::Symbol(_), Token::LCurly) => self.parse_symbol()?,
            _ => self.parse_expression()?,
        };
        self.expect(&Token::LCurly)?;

        let true_scope = self.push_scope(false);
        let mut true_stmts = Vec::new();
        while self.lexers[0].top.tok != Token::RCurly {
            true_stmts.push(self.parse_fn_stmt()?);
        }
        let mut end = self.lexers[0].top.range.end;
        self.expect(&Token::RCurly)?;
        let true_stmts = self.push_id_vec(true_stmts);

        self.pop_scope(true_scope);
        let false_scope = self.push_scope(false);
        let mut false_stmts = Vec::new();
        if self.lexers[0].top.tok == Token::Else {
            self.lexers[0].pop(); // `else`

            if self.lexers[0].top.tok == Token::If {
                false_stmts.push(self.parse_if()?);
            } else {
                self.expect(&Token::LCurly)?;

                while self.lexers[0].top.tok != Token::RCurly {
                    false_stmts.push(self.parse_fn_stmt()?);
                }

                end = self.lexers[0].top.range.end;
                self.expect(&Token::RCurly)?;
            }
        }
        let false_stmts = self.push_id_vec(false_stmts);
        self.pop_scope(false_scope);

        let range = Range::new(start, end);

        Ok(self.push_node(
            range,
            Node::If {
                cond: cond_expr,
                true_stmts,
                false_stmts,
            },
        ))
    }

    fn parse_while(&mut self, start: Location) -> Result<Id, CompileError> {
        self.expect(&Token::While)?;

        let cond_expr = match (&self.lexers[0].top.tok, &self.lexers[0].second.tok) {
            (Token::Symbol(_), Token::LCurly) => self.parse_symbol()?,
            _ => {
                if start.line == 219 && start.col == 9 {
                    let _stopme = 3;
                }
                self.parse_expression()?
            }
        };
        self.expect(&Token::LCurly)?;

        let while_scope = self.push_scope(false);
        let mut stmts = Vec::new();
        while self.lexers[0].top.tok != Token::RCurly {
            stmts.push(self.parse_fn_stmt()?);
        }
        let stmts = self.push_id_vec(stmts);
        self.pop_scope(while_scope);

        let range = self.expect_range(start, Token::RCurly)?;

        Ok(self.push_node(
            range,
            Node::While {
                cond: cond_expr,
                stmts,
            },
        ))
    }

    pub fn parse_expression(&mut self) -> Result<Id, CompileError> {
        let mut operators = Vec::new();
        let mut output = Vec::new();

        let (mut parsing_op, mut parsing_expr) = (false, true);

        loop {
            match self.lexers[0].top.tok {
                Token::EqEq
                | Token::Neq
                | Token::Add
                | Token::Sub
                | Token::Asterisk
                | Token::Div
                | Token::LAngle
                | Token::RAngle => {
                    if !parsing_op {
                        break;
                    }

                    // todo(chad): operator precedence
                    operators.push(self.lexers[0].top.tok);
                    self.lexers[0].pop();
                }
                Token::IntegerLiteral(_, _)
                | Token::FloatLiteral(_)
                | Token::StringLiteral(_)
                | Token::Ampersand
                | Token::Caret
                | Token::True
                | Token::False
                | Token::LCurly
                | Token::LSquare
                | Token::TypeOf
                | Token::LParen
                | Token::Cast
                | Token::Symbol(_)
                | Token::I8
                | Token::I16
                | Token::I32
                | Token::I64 => {
                    if !parsing_expr {
                        break;
                    }

                    let id = self.parse_expression_piece()?;
                    output.push(Shunting::Id(id))
                }
                _ => break,
            }

            std::mem::swap(&mut parsing_op, &mut parsing_expr);
        }

        while !operators.is_empty() {
            output.push(Shunting::Operator(operators.pop().unwrap()));
        }

        if output.len() == 1 {
            return Ok(match output[0] {
                Shunting::Id(id) => id,
                _ => unreachable!(),
            });
        }

        self.shunting_unroll(&mut output)
    }

    fn shunting_unroll(&mut self, output: &mut Vec<Shunting>) -> Result<Id, CompileError> {
        if output.is_empty() {
            return Err(CompileError::from_string(
                format!(
                    "Unexpected token while parsing expression: '{:?}'",
                    self.lexers[0].top.tok
                ),
                self.lexers[0].top.range,
            ));
        }

        match output.last().unwrap() {
            Shunting::Operator(Token::EqEq) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::EqEq(lhs, rhs)))
            }
            Shunting::Operator(Token::Neq) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::Neq(lhs, rhs)))
            }
            Shunting::Operator(Token::Add) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::Add(lhs, rhs)))
            }
            Shunting::Operator(Token::Sub) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::Sub(lhs, rhs)))
            }
            Shunting::Operator(Token::Asterisk) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::Mul(lhs, rhs)))
            }
            Shunting::Operator(Token::Div) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::Div(lhs, rhs)))
            }
            Shunting::Operator(Token::LAngle) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::LessThan(lhs, rhs)))
            }
            Shunting::Operator(Token::RAngle) => {
                output.pop();
                let rhs = output.pop().unwrap().as_id();
                let lhs = output.pop().unwrap().as_id();
                let range = Range::spanning(self.ranges[lhs], self.ranges[rhs]);

                Ok(self.push_node(range, Node::GreaterThan(lhs, rhs)))
            }
            Shunting::Id(id) => Err(CompileError::from_string(
                "Could not parse operator",
                self.ranges[*id],
            )),
            _ => unreachable!(),
        }
    }

    pub fn parse_expression_piece(&mut self) -> Result<Id, CompileError> {
        match self.lexers[0].top.tok {
            Token::IntegerLiteral(_, _) | Token::FloatLiteral(_) => self.parse_numeric_literal(),
            Token::StringLiteral(sym) => self.parse_string_literal(sym),
            Token::Ampersand => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `&`

                let expr = self.parse_expression()?;
                self.node_is_addressable[expr] = true;
                let end = self.ranges[expr].end;

                Ok(self.push_node(Range::new(start, end), Node::Ref(expr)))
            }
            Token::Caret => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `^`

                let expr = self.parse_expression()?;
                self.node_is_addressable[expr] = true;
                let end = self.ranges[expr].end;

                Ok(self.push_node(Range::new(start, end), Node::Load(expr)))
            }
            Token::True => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop(); // `true`
                Ok(self.push_node(range, Node::BoolLiteral(true)))
            }
            Token::False => {
                let range = self.lexers[0].top.range;
                self.lexers[0].pop(); // `false`
                Ok(self.push_node(range, Node::BoolLiteral(false)))
            }
            Token::LCurly => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `{`

                let params = self.parse_value_params(false)?;

                let range = self.expect_range(start, Token::RCurly)?;
                Ok(self.push_node(range, Node::StructLiteral { name: None, params }))
            }
            Token::LSquare => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `[`
                self.expect(&Token::RSquare)?;

                self.expect(&Token::LCurly)?;

                let params = self.parse_value_params(false)?;

                let range = self.expect_range(start, Token::RCurly)?;

                Ok(self.push_node(range, Node::ArrayLiteral(params)))
            }
            Token::TypeOf => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `#type_of`
                self.expect(&Token::LParen)?;
                let expr = self.parse_expression()?;
                let range = self.expect_range(start, Token::RParen)?;

                let id = self.push_node(range, Node::TypeOf(expr));
                self.node_is_addressable[id] = true;

                Ok(id)
            }
            Token::Cast => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `#cast`
                let expr = self.parse_expression_piece()?;

                let range = Range::new(start, self.ranges[expr].end);

                Ok(self.push_node(range, Node::Cast(expr)))
            }
            _ => {
                let start = self.lexers[0].top.range.start;

                let mut value = self.parse_lvalue()?;

                // dot / function call / macro call / array access?
                while self.lexers[0].top.tok == Token::Dot
                    || self.lexers[0].top.tok == Token::LParen
                    || self.lexers[0].top.tok == Token::LSquare
                    || self.lexers[0].top.tok == Token::DoubleLAngle
                    || self.lexers[0].top.tok == Token::Bang
                {
                    if self.lexers[0].top.tok == Token::Dot {
                        self.lexers[0].pop(); // `.`
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
                                is_assignment: false,
                            },
                        );

                        // all field accesses represent addressable memory
                        self.node_is_addressable[value] = true;
                    } else if self.lexers[0].top.tok == Token::DoubleLAngle
                        || self.lexers[0].top.tok == Token::LParen
                    {
                        let ct_params = if self.lexers[0].top.tok == Token::DoubleLAngle {
                            self.lexers[0].pop(); // `::<`
                            let params = self.parse_value_params(true)?;
                            self.expect(&Token::DoubleRAngle)?;
                            Some(params)
                        } else {
                            None
                        };

                        self.expect(&Token::LParen)?;
                        let params = self.parse_value_params(false)?;
                        let range = self.expect_range(start, Token::RParen)?;

                        value = self.push_node(
                            range,
                            Node::Call {
                                name: value,
                                ct_params,
                                params,
                            },
                        );

                        self.calls.push(value);
                    } else if self.lexers[0].top.tok == Token::LSquare {
                        self.lexers[0].pop(); // `[`
                        let index = self.parse_expression()?;
                        let range = self.expect_range(start, Token::RSquare)?;
                        value = self.push_node(range, Node::ArrayAccess { arr: value, index });
                    } else if self.lexers[0].top.tok == Token::Bang {
                        // macro invocation, expect an open paren followed by raw tokens, followed by a close paren
                        // todo(chad): expand the scope of these delimiters to more than just curlys
                        let tokens_start = self.lexers[0].top.range.start;

                        self.lexers[0].pop(); // `!`
                        self.expect(&Token::LCurly)?;

                        // todo(chad): accept more params in macro invocations?
                        let mut tokens = Vec::new();
                        while self.lexers[0].top.tok != Token::RCurly {
                            tokens.push(self.lexers[0].top);
                            self.lexers[0].pop();
                        }
                        let tokens = self.push_token_vec(tokens);

                        let range = self.expect_range(start, Token::RCurly)?;
                        let tokens_range = Range::new(tokens_start, range.end);

                        let expanded = self.push_id_vec(Vec::new());

                        value = self.push_node(
                            range,
                            Node::MacroCall {
                                name: value,
                                input: tokens,
                                expanded,
                            },
                        );

                        let cur_fun = *self.function_scopes.last().unwrap();

                        self.macro_calls.push(value);
                        self.macro_calls_by_function
                            .entry(cur_fun)
                            .or_default()
                            .push(value);
                        self.function_by_macro_call.insert(value, cur_fun);
                    } else {
                        todo!("unhandled case for parser")
                    }
                }

                Ok(value)
            }
        }
    }

    fn parse_string_literal(&mut self, sym: Sym) -> Result<Id, CompileError> {
        let range = self.lexers[0].top.range;
        self.lexers[0].pop();

        let bytes = self.lexers[0]
            .string_interner
            .resolve(sym)
            .unwrap()
            .bytes()
            .len();

        let string_id = self.push_node(range, Node::StringLiteral { sym, bytes });
        self.string_literals.push(string_id);

        // structs are always addressable (for now)
        self.node_is_addressable[string_id] = true;

        Ok(string_id)
    }

    fn parse_lvalue(&mut self) -> Result<Id, CompileError> {
        if self.lexers[0].top.tok == Token::LParen {
            self.lexers[0].pop(); // `(`
            let expr = self.parse_expression()?;
            self.expect(&Token::RParen)?;

            Ok(expr)
        } else if self.lexers[0].top.tok == Token::Caret {
            let start = self.lexers[0].top.range.start;
            self.lexers[0].pop(); // `^`

            let expr = self.parse_expression()?;
            let end = self.ranges[expr].end;

            Ok(self.push_node(Range::new(start, end), Node::Load(expr)))
        } else if self.lexers[0].top.tok == Token::I8 {
            let range = self.lexers[0].top.range;
            self.lexers[0].pop();
            Ok(self.push_node(range, Node::TypeLiteral(Type::Basic(BasicType::I8))))
        } else if self.lexers[0].top.tok == Token::I16 {
            let range = self.lexers[0].top.range;
            self.lexers[0].pop();
            Ok(self.push_node(range, Node::TypeLiteral(Type::Basic(BasicType::I16))))
        } else if self.lexers[0].top.tok == Token::I32 {
            let range = self.lexers[0].top.range;
            self.lexers[0].pop();
            Ok(self.push_node(range, Node::TypeLiteral(Type::Basic(BasicType::I32))))
        } else if self.lexers[0].top.tok == Token::I64 {
            let range = self.lexers[0].top.range;
            self.lexers[0].pop();
            Ok(self.push_node(range, Node::TypeLiteral(Type::Basic(BasicType::I64))))
        } else if let Token::Symbol(_) = self.lexers[0].top.tok {
            Ok(self.parse_symbol()?)
        } else {
            Err(CompileError::from_string(
                "Failed to parse lvalue",
                self.lexers[0].top.range,
            ))
        }
    }

    fn expect_range(&mut self, start: Location, token: Token) -> Result<Range, CompileError> {
        let range = Range::new(start, self.lexers[0].top.range.end);

        if self.lexers[0].top.tok == token {
            self.lexers[0].pop();
            Ok(range)
        } else {
            Err(CompileError::from_string(
                format!("Expected '{:?}', found {:?}", token, self.lexers[0].top.tok),
                range,
            ))
        }
    }

    pub fn parse_fn_stmt(&mut self) -> Result<Id, CompileError> {
        let start = self.lexers[0].top.range.start;

        match self.lexers[0].top.tok {
            Token::If => self.parse_if(),
            Token::While => self.parse_while(start),
            Token::Return => {
                self.lexers[0].pop(); // `return`
                let expr = self.parse_expression()?;
                let range = self.expect_range(start, Token::Semicolon)?;

                let ret_id = self.push_node(range, Node::Return(expr));

                if !self.is_in_insert {
                    self.returns.push(ret_id);
                }

                Ok(ret_id)
            }
            Token::Let => {
                self.lexers[0].pop(); // `let`
                let name = self.parse_sym()?;

                let ty = if self.lexers[0].top.tok == Token::Colon {
                    self.expect(&Token::Colon)?;
                    Some(self.parse_type()?)
                } else {
                    None
                };

                self.expect(&Token::Eq)?;

                let expr = match self.lexers[0].top.tok {
                    Token::Uninit => {
                        self.lexers[0].pop(); // pop 'uninit' token
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

                match self.lexers[0].top.tok {
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
                    _ => {
                        self.ranges[lvalue] = self.expect_range(start, Token::Semicolon)?;
                        Ok(lvalue)
                    }
                }
            }
        }
    }

    fn push_scope(&mut self, is_function_scope: bool) -> PushedScope {
        self.scopes.push(Scope::new(self.top_scope));
        let pushed = PushedScope(self.top_scope, is_function_scope);
        self.top_scope = self.scopes.len() - 1;

        if is_function_scope {
            self.function_scopes.push(self.top_scope);
        }

        pushed
    }

    fn pop_scope(&mut self, pushed: PushedScope) {
        self.top_scope = pushed.0;
        if pushed.1 {
            self.function_scopes.pop();
        }
    }

    fn expect(&mut self, tok: &Token) -> Result<(), CompileError> {
        match &self.lexers[0].top.tok {
            t if t == tok => {
                self.lexers[0].pop();
                Ok(())
            }
            _ => Err(CompileError::from_string(
                format!("Expected {:?}, got {:?}", tok, self.lexers[0].top.tok),
                self.lexers[0].top.range,
            )),
        }
    }

    fn parse_decl_params(&mut self, is_ct: bool) -> Result<IdVec, CompileError> {
        let mut params = Vec::new();

        let mut index = 0;
        while self.lexers[0].top.tok != Token::RParen
            && self.lexers[0].top.tok != Token::RCurly
            && self.lexers[0].top.tok != Token::RAngle
        {
            let input_start = self.lexers[0].top.range.start;

            let name = self.parse_symbol()?;
            let name_sym = self.nodes[name].as_symbol().unwrap();

            let ty = if self.lexers[0].top.tok == Token::Colon {
                self.lexers[0].pop();
                self.parse_type()?
            } else {
                let range = self.ranges[name];
                let params = self.push_id_vec(Vec::new());
                self.push_node(
                    range,
                    Node::TypeLiteral(Type::Struct {
                        params,
                        copied_from: None,
                    }),
                )
            };

            let range = Range::new(input_start, self.ranges[ty].end);

            // put the param in scope
            let param = self.push_node(
                range,
                Node::DeclParam {
                    name: name_sym,
                    ty,
                    index,
                    is_ct,
                    ct_link: None,
                },
            );
            self.scope_insert(name_sym, param);
            params.push(param);

            self.node_is_addressable[param] = true;

            if self.lexers[0].top.tok == Token::Comma {
                self.lexers[0].pop(); // `,`
            }

            index += 1;
        }

        Ok(self.push_id_vec(params))
    }

    pub fn push_id_vec(&mut self, id_vec: Vec<Id>) -> IdVec {
        self.id_vecs.push(id_vec);
        IdVec(self.id_vecs.len() - 1)
    }

    pub fn push_token_vec(&mut self, token_vec: Vec<Lexeme>) -> TokenVec {
        self.token_vecs.push(token_vec);
        TokenVec(self.token_vecs.len() - 1)
    }

    fn parse_value_param(&mut self, index: u16, is_ct: bool) -> Result<Id, CompileError> {
        let start = self.lexers[0].top.range.start;

        let name = if let Token::Symbol(_) = self.lexers[0].top.tok {
            if self.lexers[0].second.tok == Token::Colon {
                let name = self.parse_sym()?;
                self.lexers[0].pop(); // `;`
                Some(name)
            } else {
                None
            }
        } else {
            None
        };
        let value = self.parse_expression()?;

        let end = self.ranges[value].end;

        Ok(self.push_node(
            Range::new(start, end),
            Node::ValueParam {
                name,
                value,
                index,
                is_ct,
            },
        ))
    }

    fn parse_value_params(&mut self, is_ct: bool) -> Result<IdVec, CompileError> {
        let mut params = Vec::new();

        let mut vp_index = 0;
        while self.lexers[0].top.tok != Token::RParen
            && self.lexers[0].top.tok != Token::RCurly
            && self.lexers[0].top.tok != Token::RAngle
            && self.lexers[0].top.tok != Token::DoubleRAngle
        {
            params.push(self.parse_value_param(vp_index, is_ct)?);
            vp_index += 1;

            if self.lexers[0].top.tok == Token::Comma {
                self.lexers[0].pop(); // `,`
            }
        }

        let params = self.push_id_vec(params);

        Ok(params)
    }

    pub fn parse_fn(&mut self, start: Location) -> Result<Id, CompileError> {
        let is_macro = if self.lexers[0].top.tok == Token::Fn {
            false
        } else if self.lexers[0].top.tok == Token::Macro {
            true
        } else {
            unreachable!()
        };
        self.lexers[0].pop();

        let name = self.parse_sym()?;

        // open a new scope
        let pushed_scope = self.push_scope(true);
        self.returns.clear();

        let ct_params = if self.lexers[0].top.tok == Token::LAngle {
            self.lexers[0].pop(); // `<`
            let ct_params = self.parse_decl_params(true)?;
            self.expect(&Token::RAngle)?;
            Some(ct_params)
        } else {
            None
        };

        self.expect(&Token::LParen)?;
        let params = self.parse_decl_params(false)?;
        self.expect(&Token::RParen)?;

        let return_ty = if self.lexers[0].top.tok != Token::LCurly {
            self.parse_type()?
        } else {
            self.push_node(self.lexers[0].top.range, Node::TypeLiteral(Type::Type))
        };

        self.expect(&Token::LCurly)?;

        let mut stmts = Vec::new();
        while self.lexers[0].top.tok != Token::RCurly {
            let stmt = self.parse_fn_stmt()?;
            stmts.push(stmt);
        }
        let stmts = self.push_id_vec(stmts);

        let range = self.expect_range(start, Token::RCurly)?;

        let returns = self.push_id_vec(self.returns.clone());
        let func = self.push_node(
            range,
            Node::Func {
                name,
                scope: self.top_scope,
                ct_params,
                params,
                return_ty,
                stmts,
                returns,
                is_macro,
                copied_from: None,
            },
        );

        // pop the top scope
        // TODO: do this in some kind of 'defer'?
        self.pop_scope(pushed_scope);

        if !self.copying {
            self.scope_insert(name, func);
        }

        self.top_level_map.insert(name, func);

        if is_macro {
            self.macros.push(func);
        }

        Ok(func)
    }

    pub fn parse_repl(&mut self) -> Result<Id, CompileError> {
        match self.lexers[0].top.tok {
            Token::Import
            | Token::Struct
            | Token::Enum
            | Token::Extern
            | Token::Fn
            | Token::Macro => {
                let tl = self.parse_top_level()?;
                self.top_level.push(tl);
                Ok(tl)
            }
            _ => self.parse_expression(),
        }
    }

    pub fn parse_top_level(&mut self) -> Result<Id, CompileError> {
        let start = self.lexers[0].top.range.start;

        match self.lexers[0].top.tok {
            Token::Import => self.parse_import(start),
            Token::Struct => self.parse_struct(start),
            Token::Enum => self.parse_enum(start),
            Token::Extern => {
                let start = self.lexers[0].top.range.start;
                self.lexers[0].pop(); // `extern`

                let name = self.parse_sym()?;

                self.expect(&Token::LParen)?;
                let params = self.parse_decl_params(false)?;
                self.expect(&Token::RParen)?;

                let return_ty = self.parse_type()?;

                let range = self.expect_range(start, Token::Semicolon)?;

                let ext_id = self.push_node(
                    range,
                    Node::Extern {
                        name,
                        params,
                        return_ty,
                    },
                );
                self.scope_insert(name, ext_id);
                self.externs.push(ext_id);
                self.top_level.push(ext_id);

                Ok(ext_id)
            }
            Token::Fn | Token::Macro => self.parse_fn(start),
            _ => self.parse_expression(),
            // _ => Err(CompileError::from_string(
            //     "Unexpected top level",
            //     self.lexers[0].top.range,
            // )),
        }
    }

    pub fn parse_import(&mut self, start: Location) -> Result<Id, CompileError> {
        self.lexers[0].pop();
        let name = match self.lexers[0].top.tok {
            Token::StringLiteral(sym) => self.parse_string_literal(sym),
            _ => Err(CompileError::from_string(
                "Expected string literal as import target",
                Range::new(start, self.lexers[0].top.range.end),
            )),
        }?;
        let name = self.nodes[name].as_string_literal().unwrap();

        let range = self.expect_range(start, Token::Semicolon)?;
        Ok(self.push_node(range, Node::Import { name }))
    }

    pub fn parse_struct(&mut self, start: Location) -> Result<Id, CompileError> {
        self.lexers[0].pop();
        let name = self.parse_sym()?;

        let ct_params = if self.lexers[0].top.tok == Token::LAngle {
            self.lexers[0].pop(); // `<`
            let params = self.parse_decl_params(true)?;
            self.expect(&Token::RAngle)?;
            Some(params)
        } else {
            None
        };

        self.expect(&Token::LCurly)?;
        let params = self.parse_decl_params(false)?;
        let range = self.expect_range(start, Token::RCurly)?;

        let id = self.push_node(
            range,
            Node::Struct {
                name,
                ct_params,
                params,
            },
        );

        if !self.copying {
            self.scope_insert(name, id);
        }

        Ok(id)
    }

    pub fn parse_enum(&mut self, start: Location) -> Result<Id, CompileError> {
        self.lexers[0].pop();
        let name = self.parse_sym()?;

        let ct_params = if self.lexers[0].top.tok == Token::LAngle {
            self.lexers[0].pop(); // `<`
            let params = self.parse_decl_params(true)?;
            self.expect(&Token::RAngle)?;
            Some(params)
        } else {
            None
        };

        self.expect(&Token::LCurly)?;
        let params = self.parse_decl_params(false)?;
        let range = self.expect_range(start, Token::RCurly)?;

        let id = self.push_node(
            range,
            Node::Enum {
                name,
                ct_params,
                params,
            },
        );

        if self.ty_decl == None && self.resolve_sym_unchecked(name) == "Ty" {
            self.ty_decl = Some(id);
        }

        if !self.copying {
            self.scope_insert(name, id);
        }

        Ok(id)
    }

    pub fn debug(&self, id: Id) -> String {
        let range = self.ranges[id];

        self.lexers[0].source_info.original_source[range.start.char_offset..range.end.char_offset]
            .to_string()
    }

    pub fn id_vec(&self, idv: IdVec) -> &Vec<Id> {
        &self.id_vecs[idv.0]
    }

    #[allow(dead_code)]
    pub fn id_vec_mut(&mut self, idv: IdVec) -> &mut Vec<Id> {
        &mut self.id_vecs[idv.0]
    }
}

#[derive(Copy, Clone, Debug)]
enum Shunting {
    Operator(Token),
    Id(Id),
}

impl Shunting {
    fn as_id(self) -> Id {
        match self {
            Self::Id(id) => id,
            _ => panic!("Expected Shunting::Id"),
        }
    }
}
