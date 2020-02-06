#[macro_use]
extern crate lazy_static;

#[macro_use(defer)]
extern crate scopeguard;

mod backend;
mod parser;
mod semantic;

use backend::Backend;

fn main() {
    let source = std::fs::read_to_string("src/foo.flea").unwrap();
    let mut backend = Backend::bootstrap_from_source(&source)
        .map_err(|e| panic!("Error bootstrapping from source: {:?}", e))
        .unwrap();
    dbg!(backend.call_func("main"));

    //     backend
    //         .update_source(
    //             "
    // (fn main () i64
    //     (let a bool (call bar))
    //     (let b i64 (if a 3 5))
    //     (return b))

    // (fn bar () bool
    //     (return false))
    //     ",
    //         )
    //         .map_err(|e| panic!("Error bootstrapping from source: {:?}", e))
    //         .unwrap();

    //     backend
    //         .recompile_function("bar")
    //         .map_err(|e| panic!("Error recompiling function: {:?}", e))
    //         .unwrap();
    //     dbg!(backend.call_func("main"));
}
