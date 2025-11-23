#!/usr/bin/env bash

set -euxo pipefail

cargo clippy --fix --allow-dirty
cargo fmt
cargo audit fix
cargo fix --all-targets --all-features --allow-dirty
