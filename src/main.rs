#[macro_use]
extern crate lazy_static;

mod backend;
mod parser;
mod semantic;

use backend::Backend;
use parser::CompileError;

fn main() -> Result<(), CompileError> {
    let source = std::fs::read_to_string("src/foo.cpi").unwrap();

    let mut semantic = Backend::bootstrap_to_semantic(&source)?;
    let backend = Backend::bootstrap_to_backend(&mut semantic)?;

    // println!("**** RUNTIME ****");
    // let now = std::time::Instant::now();
    dbg!(backend.call_func("main"));
    // let elapsed = now.elapsed();
    // println!(
    //     "Function 'main' ran in {}",
    //     elapsed.as_micros() as f64 / 1_000_000.0
    // );

    // backend.update_source_at_top_level(
    //     "
    // fn foo() i64 {
    //     return 8;
    // }
    //             ",
    //     Location {
    //         line: 189,
    //         col: 0,
    //         char_offset: 0,
    //     },
    // )?;
    // backend.recompile_function("foo")?;

    // dbg!(backend.call_func("main"));

    Ok(())
}
