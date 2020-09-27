# TODO

## Definitely Implement

- [ ] module system
    - simple namespacing
    - for 
- [ ] more compiler info for a given node (similar to #type_of info)
    - [ ] basically, this boils down to exposing the backend/semantic/parser/lexer infos with some collection of extern functions. Then, all we have to do is have a function to get the id of a node and then everything should be available
- [ ] macros should not recieve an array of tokens. Instead, they should receive a node id.
    - the stream of tokens (as well as lots of other good info) can be had from the compiler
- [ ] global values
    - Declaring things at top-level is tricky. For example, what happens if someone declares a struct type, then adds a top-level value of that type, then changes the struct's type definition? What happens if there are functions that reference the struct type in the argument list or function body?
        - one strategy is to add a prelude to every struct member access to ensure the type is what you think it is. If not, then some kind of a panic/exception could be raised. Obviously this could be disabled for performance reasons if desired.
        - another strategy would be to provide two ways of recompiling -- recompile only function/type/extern declarations, and recompile everything (including top-level statements about global values).
- [ ] garbage collection
    - concurrent, parallel, non-moving tricolor mark/sweep
- [ ] integer promotions
- [ ] structs as return values
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

- [ ] baked functions (`let f = poly<i64>;` and then use f as a function pointer)
- [ ] baked function values (making a version of a function where one or more parameters are compile-time known)
