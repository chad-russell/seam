#[macro_use]
extern crate lazy_static;

extern crate crossterm;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent},
    style::{self, Colorize},
    QueueableCommand,
};

mod backend;
mod parser;
mod semantic;

use std::io::{stdout, Write};

use backend::{Backend, Value, FUNC_PTRS};
use parser::{CompileError, Node, Range, Token, Type};

fn main() -> Result<(), CompileError> {
    // repl().map_err(|e| CompileError::from_string(format!("{:?}", e), Range::default()))?;
    // Ok(())

    run_file()
}

fn repl() -> Result<(), CompileError> {
    let mut semantic = Backend::bootstrap_to_semantic(String::from("seam_repl"), "")?;
    let mut backend = Backend::new(&mut semantic);
    let mut repl_stmt_count = 0;

    loop {
        match repl_loop(&mut backend, repl_stmt_count) {
            Err(e) => println!("{:?}", e),
            _ => (),
        }
        repl_stmt_count += 1;
    }
}

fn repl_loop(backend: &mut Backend, repl_stmt_count: i64) -> Result<(), CompileError> {
    prompt().map_err(|e| CompileError::from_string(e.to_string(), Range::default()))?;
    let r = read_line().map_err(|e| CompileError::from_string(e.to_string(), Range::default()))?;

    backend.append_source(&format!("{}\n", r));

    match backend.semantic.parser.lexer.top.tok {
        Token::Struct | Token::Enum | Token::Extern | Token::Fn | Token::Macro => {
            let tl = backend.semantic.parser.parse_top_level()?;
            backend.semantic.parser.top_level.push(tl);

            while backend.semantic.types.len() < backend.semantic.parser.nodes.len() {
                backend.semantic.types.push(Type::Unassigned);
            }
            while backend.values.len() < backend.semantic.parser.nodes.len() {
                backend.values.push(Value::Unassigned);
            }

            // TODO(chad): this ends up re-compiling everything. Be more judicious
            for id in backend.semantic.parser.top_level.clone() {
                backend.semantic.assign_type(id)?;
                backend.semantic.unify_types()?;
                backend.compile_id(id)?;
            }

            println!(
                "Available Functions:\n{}",
                FUNC_PTRS
                    .lock()
                    .unwrap()
                    .keys()
                    .map(|k| backend.semantic.parser.lexer.resolve_unchecked(*k))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        _ => {
            backend.semantic.parser.parse_expression()?;

            // TODO(chad): what's the best way to do this?
            let repl_fn_name = format!("__repl_{}", repl_stmt_count);
            backend.append_source(&format!("fn {}() {{\n return {}; \n}}\n", &repl_fn_name, r));

            let tl = backend.semantic.parser.parse_top_level()?;
            backend.semantic.parser.top_level.push(tl);

            while backend.semantic.types.len() < backend.semantic.parser.nodes.len() {
                backend.semantic.types.push(Type::Unassigned);
            }
            while backend.values.len() < backend.semantic.parser.nodes.len() {
                backend.values.push(Value::Unassigned);
            }

            // TODO(chad): this ends up re-compiling everything. Be more judicious
            for id in backend.semantic.parser.top_level.clone() {
                backend.semantic.assign_type(id)?;
                backend.semantic.unify_types()?;
                backend.compile_id(id)?;
            }

            println!("{}", backend.call_func(&repl_fn_name));
        }
    };

    Ok(())
}

pub fn prompt() -> crossterm::Result<()> {
    let mut stdout = stdout();
    stdout.queue(style::PrintStyledContent("seam> ".green()))?;
    stdout.flush()?;

    Ok(())
}

pub fn read_line() -> crossterm::Result<String> {
    let mut line = String::new();
    while let Event::Key(KeyEvent { code, .. }) = event::read()? {
        match code {
            KeyCode::Enter => {
                break;
            }
            KeyCode::Char(c) => {
                line.push(c);
            }
            _ => {}
        }
    }

    return Ok(line);
}

pub fn run_file() -> Result<(), CompileError> {
    let source = std::fs::read_to_string("tests/tests.sm").unwrap();

    let mut semantic = Backend::bootstrap_to_semantic(String::from("tests/tests"), &source)?;
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
