# TODO

## Definitely Implement

- [ ] type_of
- [ ] routine to push token from a macro
- [ ] debugging
- [ ] basic enum support
  - [ ] support `#tag_of()`
  - [ ] check enum tag when accessing through a field
- [ ] defer
- [ ] passing named params out of order, default params, unnamed params for structs
- [ ] auto-declaration of poly variables -> inline anonymous functions -> lambdas
- [ ] math (binary operations)
- [ ] if
  - [ ] as value expression
  - [ ] `else if`
- [ ] short-circuit logical `and` and `or`
- [ ] while
- [ ] for (with macros?)
- [ ] f32/f64
- [ ] structs as function parameters / return values
  - [ ] follow C ABI, at least for extern functions

## Maybe Implement

- [ ] compile-time function calls `#run()`
- [ ] baked struct types (`let p: Poly<bool> = ---;`)
- [ ] baked functions (`let f = poly<i64>;` and then use f as a function pointer)
- [ ] baked funciton values (making a version of a function where one or more parameters are compile-time known)
