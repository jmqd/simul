{
  description = "Rust dev environment";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixpkgs-intel-darwin.url = "github:NixOS/nixpkgs/nixpkgs-26.05-darwin";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-intel-darwin,
      rust-overlay,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];

        # 26.05 is the last supported nixpkgs release for Intel macOS.
        nixpkgsForSystem = if system == "x86_64-darwin" then nixpkgs-intel-darwin else nixpkgs;
        pkgs = import nixpkgsForSystem { inherit system overlays; };

        rustToolchainConfig = builtins.fromTOML (builtins.readFile ./rust-toolchain.toml);
        rustToolchainChannel = rustToolchainConfig.toolchain.channel;

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        rustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        dylintVersion = "6.2.0";
        dylintSource = pkgs.fetchFromGitHub {
          owner = "trailofbits";
          repo = "dylint";
          rev = "v${dylintVersion}";
          hash = "sha256-UAYzmjRNzc19ztXIGDzresbWoVH7CYgq9b1/fp4bjZs=";
        };
        dylintCargoHash = "sha256-3Yv4mdpuvNga34L12a3k7N8twJ/WAbvF0uhK+/hVWPU=";

        cargoDylint = rustPlatform.buildRustPackage {
          pname = "cargo-dylint";
          version = dylintVersion;
          src = dylintSource;
          cargoHash = dylintCargoHash;
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];
          buildAndTestSubdir = "cargo-dylint";
          # Use the exact published driver, not the temporary Nix build source path.
          buildFeatures = [ "dylint/__driver_from_crates_io" ];
          doCheck = false;
        };

        dylintLink = rustPlatform.buildRustPackage {
          pname = "dylint-link";
          version = dylintVersion;
          src = dylintSource;
          cargoHash = dylintCargoHash;
          nativeBuildInputs = [ pkgs.pkg-config ];
          buildInputs = [ pkgs.openssl ];
          buildAndTestSubdir = "dylint-link";
          doCheck = false;
        };

        # Dylint asks `rustup +stable which cargo` while resolving lint libraries.
        # The dev shell's pinned toolchain is newer than the user's rustup `stable`,
        # so route that lookup back to the pinned Cargo and delegate all other
        # rustup operations normally.
        dylintRustup = pkgs.writeShellScriptBin "rustup" ''
          if [ "$#" -eq 3 ] && [ "$1" = "+stable" ] && [ "$2" = "which" ] && [ "$3" = "cargo" ]; then
            echo "${rustToolchain}/bin/cargo"
          else
            exec ${pkgs.rustup}/bin/rustup "$@"
          fi
        '';

        # Keep this checkout on the pinned Nix toolchain. Nested projects that explicitly
        # request another channel still delegate to the user's rustup installation.
        dylintCargo = pkgs.writeShellScriptBin "cargo" ''
          find_toolchain_dir() {
            dir="$PWD"
            while [ "$dir" != "/" ]; do
              if [ -f "$dir/rust-toolchain.toml" ] || [ -f "$dir/rust-toolchain" ]; then
                printf '%s\n' "$dir"
                return
              fi
              dir="$(${pkgs.coreutils}/bin/dirname "$dir")"
            done
          }

          toolchain="''${RUSTUP_TOOLCHAIN:-}"
          if [ -z "$toolchain" ]; then
            toolchain_dir="$(find_toolchain_dir || true)"
            if [ -n "$toolchain_dir" ]; then
              toolchain_file="$toolchain_dir/rust-toolchain"
              if [ -f "$toolchain_dir/rust-toolchain.toml" ]; then
                toolchain_file="$toolchain_dir/rust-toolchain.toml"
              fi

              toolchain="$(${pkgs.gnused}/bin/sed -n 's/^[[:space:]]*channel[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$toolchain_file" | ${pkgs.coreutils}/bin/head -n 1)"
              if [ -z "$toolchain" ]; then
                toolchain="$(${pkgs.coreutils}/bin/head -n 1 "$toolchain_file")"
              fi
            fi
          fi

          if [ -n "$toolchain" ] && [ "$toolchain" != "${rustToolchainChannel}" ]; then
            cargo_path="$(${pkgs.rustup}/bin/rustup which --toolchain "$toolchain" cargo)"
            toolchain_bin="$(${pkgs.coreutils}/bin/dirname "$cargo_path")"
            toolchain_root="$(${pkgs.coreutils}/bin/dirname "$toolchain_bin")"
            toolchain_name="$(${pkgs.coreutils}/bin/basename "$toolchain_root")"
            exec ${pkgs.coreutils}/bin/env \
              PATH="$toolchain_bin:$PATH" \
              RUSTC="$toolchain_bin/rustc" \
              RUSTDOC="$toolchain_bin/rustdoc" \
              RUSTUP_TOOLCHAIN="$toolchain_name" \
              "$cargo_path" "$@"
          fi

          exec ${rustToolchain}/bin/cargo "$@"
        '';

        simul = rustPlatform.buildRustPackage {
          pname = "simul";
          version = "0.5.1";
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
          };
          nativeBuildInputs = [ ];
          buildInputs = [ ];
        };
      in
      {
        packages.default = simul;
        apps.default = flake-utils.lib.mkApp { drv = simul; };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            dylintCargo
            rustToolchain
            openssl
            zlib
            pkg-config
            dylintRustup
            git
            llvmPackages.bolt

            # Cargo checks / lints / tools
            cargoDylint
            dylintLink
            cargo-audit
            cargo-deny
            cargo-edit
            cargo-license
            cargo-pgo
            cargo-udeps
            cargo-watch
            just
          ];

          shellHook = ''
            # Tells rust-analyzer where the stdlib sources are
            export RUST_SRC_PATH=${rustToolchain}/lib/rustlib/src/rust/library
          '';
        };
      }
    );
}
