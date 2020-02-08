#[macro_use]
extern crate lazy_static;

#[macro_use(defer)]
extern crate scopeguard;

mod backend;
mod parser;
mod semantic;

use backend::Backend;
use parser::CompileError;

fn main() -> Result<(), CompileError> {
    let source = std::fs::read_to_string("src/foo.flea").unwrap();
    let mut backend = Backend::bootstrap_from_source(&source)?;
    dbg!(backend.call_func("main"));

    //     backend.update_source(
    //         "
    // fn main() i64 {
    //     let a: i64 = ---;
    //     a = bar();
    //     return a;
    // }

    // fn bar() i64 {
    //     return 18;
    // }
    //         ",
    //     )?;

    //     backend.recompile_function("bar")?;
    //     dbg!(backend.call_func("main"));

    Ok(())
}
