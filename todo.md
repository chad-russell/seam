# TODO

- [x] local variables
- [x] if/else
- [x] functions as values
- [x] pointers
  - [x] address-of
  - [x] permit taking address of immediate value -- do the trick where we put it on the stack and then save it so it has memory
  - [x] deref
  - [x] function pointers
- [x] function parameters
- [x] structs
  - [x] declaration
  - [x] struct literals (with own slot)
  - [x] dot syntax
  - [x] address of field
- [x] syntax overhaul -- more c/rust/go/jai like

- [ ] type inference
- [ ] generics

- [ ] enum (??)

    ```rust
        // enum Option(T: Type) {
        //     Some(T),
        //     None,
        // }

        enum Option {
            some: $T,
            none: unit,
        }

        fn test_option() {
            let opt1: Option(T: i64) = Option(T: i64) { some: 3 };
            let opt2: Option = Option(i64) { some: 3 };
            let opt6 = Option { some: 3 };
            let opt4 = Option(i64) { none: --- };

            if opt1.tag == tag!(Option, some);
            if tag_is!(opt2, some) {
                do_stuff(opt2.some);
            }
        }
    ```

- [ ] void return type
- [ ] single function call as a statement (i.e. `foo();` should be a valid function statement)
- [ ] structs as function parameters / return values
  - [ ] follow C ABI, at least for extern functions
- [ ] arrays
- [ ] while
- [ ] f32/f64
- [ ] more math
