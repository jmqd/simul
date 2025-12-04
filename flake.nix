{
  description = "Rust dev environment";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];

        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain =
          pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        simul = rustPlatform.buildRustPackage {
          pname = "simul";
          version = "0.4.1";
          src = ./.;
          cargoLock = { lockFile = ./Cargo.lock; };
          nativeBuildInputs = [ ];
          buildInputs = [ ];
        };
      in {
        packages.default = simul;
        apps.default = flake-utils.lib.mkApp { drv = simul; };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustToolchain
            openssl

            # Cargo checks / lints / tools
            cargo-audit
            cargo-deny
            cargo-edit
            cargo-license
            cargo-pgo
            cargo-udeps
            cargo-watch
          ];

          shellHook = ''
            # Tells rust-analyzer where the stdlib sources are
            export RUST_SRC_PATH=${rustToolchain}/lib/rustlib/src/rust/library
          '';
        };
      });
}
