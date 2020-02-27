# TODO

## Definitely Implement

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

## Optimization

- [ ] Instead of deep-copying the whole function, deep-copy only the signature (ct-params & params) of polymorphic functions. Basically copy as if the function had no body
      Then, when doing semantic analysis see if we can completely specialize the function with no body.
      - If yes, then see if there's already an entry. 
        - If there is we are done, else see next line
      - If no, then copy the body, then do codegen and see if there's an entry

## Maybe Implement

- [ ] compile-time function calls `#run()`
- [ ] baked struct types (`let p: Poly<bool> = ---;`)
- [ ] baked functions (`let f = poly<i64>;` and then use f as a function pointer)
- [ ] baked funciton values (making a version of a function where one or more parameters are compile-time known)
