#!/usr/bin/env bash

set -euxo pipefail

cargo check --all-features --all-targets
cargo test --all-features --all-targets
cargo fmt --check --all
cargo clippy --all-features --all-targets -- -D warnings
cargo audit --all-targets
cargo doc --all-features --all-targets

licenses_outside_allowlist=$(cargo license | grep -Ev "((MIT)|(Apache-2.0)|(BSD-[23]))")
if [ -n "$licenses_outside_allowlist" ]; then
    echo -e "\033[31m ERROR: Disallowed licenses detected:\033[0m"
    echo -e "\033[31m$licenses_outside_allowlist\033[0m"
    exit 1;
fi

RUSTC_BOOTSTRAP=1 cargo udeps
