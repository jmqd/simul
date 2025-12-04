#!/usr/bin/env bash

cargo pgo build
cargo pgo bench
cargo pgo optimize
cargo pgo bolt build
cargo pgo bolt optimize
