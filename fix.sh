#!/usr/bin/env bash

set -euxo pipefail

cargo clippy --fix --allow-dirty
cargo fmt
