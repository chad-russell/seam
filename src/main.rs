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
    // fn r3() i64 {
    //     return 8;
    // }

    // fn main() i64 {
    //     let ppf: **fn() i64 = &&r3;
    //     let pf: *fn() i64 = ^ppf;
    //     return (^pf)();
    // }
    //             ",
    //     )?;

    //     backend.recompile_function("r3")?;
    //     dbg!(backend.call_func("main"));

    Ok(())
}
