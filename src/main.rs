#[macro_use]
extern crate lazy_static;

mod backend;
mod parser;
mod semantic;

use backend::Backend;

fn main() {
    let source = std::fs::read_to_string("src/foo.flea").unwrap();
    let backend = Backend::bootstrap_from_source(&source)
        .map_err(|e| panic!("Error bootstrapping from source: {:?}", e))
        .unwrap();
    dbg!(backend.call_func("main"));

    // backend
    //     .update_source(
    //         "
    // (fn foo () i64
    //     (return (+ 101 (call baz))))

    // (fn main () i64
    //     (return (+ (call bar) 1)))

    // (fn baz () i64
    //     (return 1))

    // (fn bar () i64
    //     (return (+ 1 (call foo))))
    //     ",
    //     )
    //     .map_err(|e| panic!("Error bootstrapping from source: {:?}", e))
    //     .unwrap();

    // backend
    //     .recompile_function("foo")
    //     .map_err(|e| panic!("Error recompiling function: {:?}", e))
    //     .unwrap();
    // dbg!(backend.call_func("main"));
}
