# TODO

## Definitely Implement

- [x] structs as function parameters 
- [ ] structs as return values
- [ ] integer promotions
- [ ] import other files / modules
- [ ] switch on enums
    - [ ] check cases are exhaustive
- [ ] debugging
- [ ] f32/f64
- [ ] defer
- [ ] check enum tag when accessing through a field
- [ ] array bounds checking
- [ ] passing named params out of order, default params, unnamed params for structs
- [ ] auto-declaration of poly variables -> inline anonymous functions -> lambdas
- [ ] if as value expression
- [ ] short-circuit logical `and` and `or`
- [ ] for (with macros? or traits?)
- [ ] annotation for functions to follow C ABI
- [ ] unsigned integers
- [ ] handle empty / no return type
- [ ] traits (?)

## Optimization

- [ ] Instead of deep-copying the whole function, deep-copy only the signature (ct-params & params) of polymorphic functions. Basically copy as if the function had no body
      Then, when doing semantic analysis see if we can completely specialize the function with no body.
      - If yes and there's already an entry, we are done
      - If no, then copy the body, then do codegen and see if there's an entry


## Maybe Implement

- [ ] compile-time function calls `#run()`
- [ ] baked functions (`let f = poly<i64>;` and then use f as a function pointer)
- [ ] baked funciton values (making a version of a function where one or more parameters are compile-time known)
