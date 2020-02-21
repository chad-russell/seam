#[macro_use]
extern crate lazy_static;

mod backend;
mod parser;
mod semantic;

use backend::Backend;
use parser::CompileError;

fn main() -> Result<(), CompileError> {
    let source = std::fs::read_to_string("src/foo.flea").unwrap();

    let mut semantic = Backend::bootstrap_to_semantic(&source)?;
    let backend = Backend::bootstrap_to_backend(&mut semantic)?;

    println!("**** RUNTIME ****");
    let now = std::time::Instant::now();
    dbg!(backend.call_func("main"));
    let elapsed = now.elapsed();
    println!(
        "Function 'main' ran in {}",
        elapsed.as_micros() as f64 / 1_000_000.0
    );

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
