# List available recipes
default:
    @just --list

# Run all mandatory checks (test, clippy, fmt, audit, doc, licenses, udeps)
check: test lint fmt-check audit doc check-licenses check-udeps

# Fix all auto-fixable issues
fix: fix-clippy fix-fmt fix-audit fix-cargo

# Run tests
test:
    cargo test --all-features --all-targets

# Run clippy lints
lint:
    cargo clippy --all-features --all-targets -- -D warnings

# Install the nightly components required by the pinned Dylint libraries
dylint-toolchain:
    rustup toolchain install nightly-2026-04-16 --component rust-src --component rustc-dev --component llvm-tools-preview

# Run Dylint lints
dylint: dylint-toolchain
    CARGO_NET_GIT_FETCH_WITH_CLI=true cargo dylint --all -- --all-features --all-targets

# Check formatting
fmt-check:
    cargo fmt --check --all

# Run cargo audit
audit:
    cargo audit

# Build documentation
doc:
    cargo doc --all-features

# Check licenses (logic ported from check.sh)
check-licenses:
    @echo "Checking licenses..."
    @# The grep command will exit with 0 if it finds matching lines (bad licenses),
    @# and 1 if it finds no matches (all licenses are good).
    @# We want to fail (exit 1) if grep finds matches.
    @if cargo license | grep -v -E "((MIT)|(Apache-2.0)|(BSD-[23]))"; then \
        printf "\033[31m ERROR: Disallowed licenses detected above\033[0m\n"; \
        exit 1; \
    else \
        echo "License check passed."; \
    fi

# Check unused dependencies (requires nightly/RUSTC_BOOTSTRAP)
check-udeps:
    RUSTC_BOOTSTRAP=1 cargo udeps

# Fix clippy issues
fix-clippy:
    cargo clippy --all-targets --all-features --fix --allow-dirty

# Format code
fix-fmt:
    cargo fmt --all

# Fix audit issues
fix-audit:
    cargo audit fix

# Run cargo fix
fix-cargo:
    cargo fix --all-targets --all-features --allow-dirty

# Run benchmarks
bench:
    cargo bench
