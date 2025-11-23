#!/usr/bin/env bash

set -euxo pipefail

cargo check --all-features --all-targets
cargo test --all-features --all-targets
cargo fmt --check --all
cargo clippy --all-features --all-targets -- -D warnings
cargo audit
cargo doc --all-features

licenses_outside_allowlist=$(cargo license | grep -Ev "((MIT)|(Apache-2.0)|(BSD-[23]))") || true;
if [ -n "$licenses_outside_allowlist" ]; then
    echo -e "\033[31m ERROR: Disallowed licenses detected:\033[0m"
    echo -e "\033[31m$licenses_outside_allowlist\033[0m"
    exit 1;
fi

RUSTC_BOOTSTRAP=1 cargo udeps

# consider:
#   https://github.com/cpg314/cargo-workspace-unused-pub
#   cargo mutants --test-tool=nextest
#   RUSTDOCFLAGS="--show-coverage" cargo +nightly doc
