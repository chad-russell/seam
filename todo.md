# TODO

## Definitely Implement

- [ ] `\n` in string literal for escaping
- [ ] specification prefixes on integers (i.e. 0i64, 0i32, etc).
- [ ] routine to push token from a macro
- [ ] `else if`
- [ ] f32/f64
- [ ] `if let a.b` OR `if let my_rename = a.b`
- [ ] debugging
- [ ] check enum tag when accessing through a field
- [ ] defer
- [ ] passing named params out of order, default params, unnamed params for structs
- [ ] auto-declaration of poly variables -> inline anonymous functions -> lambdas
- [ ] if as value expression
- [ ] short-circuit logical `and` and `or`
- [ ] for (with macros?)
- [ ] structs as function parameters / return values
- [ ] annotation for functions to follow C ABI
- [ ] unsigned integers

## Maybe Implement

- [ ] compile-time function calls `#run()`
- [ ] baked struct types (`let p: Poly<bool> = ---;`)
- [ ] baked functions (`let f = poly<i64>;` and then use f as a function pointer)
- [ ] baked funciton values (making a version of a function where one or more parameters are compile-time known)
