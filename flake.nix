{
  description = "Kids ML Lab — dev shell with the native libs that Python wheels expect";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      forAllSystems = f:
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ]
          (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.uv pkgs.git ];

          # Python wheels (numpy, torch, ...) link against these at runtime.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            export KIDSML_SKIP_NIX_LIBS=1
            echo "🧪 Kids ML Lab — try:  ./run.sh app"
          '';
        };
      });
    };
}
