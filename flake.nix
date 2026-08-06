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
      devShells = forAllSystems (pkgs:
        let
          # Must satisfy `requires-python` in pyproject.toml and match `.python-version`.
          python = pkgs.python313;

          # Left to itself, uv downloads a standalone CPython built for generic Linux.
          # NixOS has no /lib64/ld-linux-x86-64.so.2 to run it with, so the first `uv run`
          # dies with "Could not start dynamically linked executable" — and no amount of
          # LD_LIBRARY_PATH helps, because what is missing is the ELF *interpreter*, not a
          # library. Hand uv this shell's interpreter instead.
          #
          # Linux only. On macOS uv's own downloads work fine, and forcing a different
          # interpreter there would invalidate everyone's existing .venv for no reason.
          useShellPython = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
            export UV_PYTHON_DOWNLOADS=never
            export UV_PYTHON=${python}/bin/python3.13

            # A .venv built against a different interpreter cannot be reused, and the error
            # when it is tried is not obvious. Say so plainly instead.
            if [ -x .venv/bin/python ] \
               && [ "$(readlink -f .venv/bin/python)" != "$(readlink -f "$UV_PYTHON")" ]; then
              echo "⚠️  .venv was built with a different Python. Run:  rm -rf .venv"
            fi
          '';
        in
        {
          default = pkgs.mkShell {
            packages = [ pkgs.uv pkgs.git ] ++ pkgs.lib.optional pkgs.stdenv.isLinux python;

            # Python wheels (numpy, torch, ...) link against these at runtime.
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ];

            shellHook = ''
              export KIDSML_SKIP_NIX_LIBS=1
              ${useShellPython}
              echo "🧪 Kids ML Lab — try:  ./run.sh app"
            '';
          };
        });
    };
}
