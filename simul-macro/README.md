# simul-macro

This crate contains the macros that are used in the `simul` crate.

This macros are re-exported via the `simul` lib and are only intended to be used
with that library, so there's no reason to take a direct dependency on this
crate.  The only reason it is a separate crate is because procedural macros must
be in a separate crate.
