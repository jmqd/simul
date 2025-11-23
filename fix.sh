#!/usr/bin/env bash

set -euxo pipefail

cargo clippy --all-targets --all-features --fix --allow-dirty
cargo fmt --all
cargo audit fix
cargo fix --all-targets --all-features --allow-dirty
