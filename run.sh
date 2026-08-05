#!/usr/bin/env bash
# Kids ML Lab launcher.
#   ./run.sh app   -> the interactive Streamlit playground (start here)
#   ./run.sh lab   -> JupyterLab with the chapter notebooks
#   ./run.sh test  -> smoke tests
#   ./run.sh build -> regenerate notebooks/*.ipynb from notebooks/_src/*.py
set -euo pipefail
cd "$(dirname "$0")"

# On NixOS, pip wheels can't find libstdc++/libz because there is no global /usr/lib.
# Point LD_LIBRARY_PATH at the right store paths (cached, since `nix eval` is slow).
if [ -e /etc/NIXOS ] && [ -z "${KIDSML_SKIP_NIX_LIBS:-}" ]; then
  cache=".nix-libs"
  if [ ! -s "$cache" ]; then
    {
      nix eval --raw nixpkgs#stdenv.cc.cc.lib
      printf '/lib:'
      nix eval --raw nixpkgs#zlib
      printf '/lib'
    } > "$cache"
  fi
  export LD_LIBRARY_PATH="$(cat "$cache")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

case "${1:-app}" in
  app)  exec uv run streamlit run app/Home.py ;;
  lab)  exec uv run jupyter lab notebooks ;;
  test) exec uv run pytest tests -q ;;
  build)
    shift
    exec uv run python tools/build_notebooks.py "$@"
    ;;
  *)    echo "usage: ./run.sh [app|lab|test|build]" >&2; exit 1 ;;
esac
