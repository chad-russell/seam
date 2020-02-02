#[macro_use]
extern crate lazy_static;

mod backend;
mod parser;
mod semantic;

use backend::Backend;
use parser::Parser;
use semantic::Semantic;

fn main() {
    let source = std::fs::read_to_string("src/foo.flea").unwrap();
    let mut parser = Parser::new(&source);
    parser
        .parse()
        .map_err(|e| panic!("Parser error: {:?}", e))
        .unwrap();

    let mut semantic = Semantic::new(parser);
    semantic
        .assign_top_level_types()
        .map_err(|e| panic!("Semantic error: {:?}", e))
        .unwrap();

    let mut backend = Backend::new(semantic);
    backend
        .compile()
        .map_err(|e| panic!("Backend error: {:?}", e))
        .unwrap();

    let main_ret = backend.call_func_no_args_i32_return("main");
    dbg!(main_ret);
}
