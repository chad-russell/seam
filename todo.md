# TODO

## Definitely Implement

- [ ] finish println
- [ ] routine to push token from a macro
- [ ] debugging
- [ ] check enum tag when accessing through a field
- [ ] defer
- [ ] passing named params out of order, default params, unnamed params for structs
- [ ] auto-declaration of poly variables -> inline anonymous functions -> lambdas
- [ ] if as value expression
- [ ] `else if`
- [ ] short-circuit logical `and` and `or`
- [ ] for (with macros?)
- [ ] f32/f64
- [ ] structs as function parameters / return values
- [ ] annotation for functions to follow C ABI

## Maybe Implement

- [ ] compile-time function calls `#run()`
- [ ] baked struct types (`let p: Poly<bool> = ---;`)
- [ ] baked functions (`let f = poly<i64>;` and then use f as a function pointer)
- [ ] baked funciton values (making a version of a function where one or more parameters are compile-time known)
