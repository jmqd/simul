#!/usr/bin/env bash

set -euxo pipefail

cargo check
cargo test
cargo fmt --check
cargo clippy -- -D warnings
cargo audit
RUSTC_BOOTSTRAP=1 cargo udeps
