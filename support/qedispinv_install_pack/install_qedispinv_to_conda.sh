#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="${ENV_NAME:-adsurf_epvr}"
SRC_ROOT="${SRC_ROOT:-$HOME/.cache/adsurf-slope-normalized-loss/external_src}"
QEDISPINV_URL="${QEDISPINV_URL:-https://github.com/pan3rock/QEDispInv.git}"
QEDISPINV_REF="${QEDISPINV_REF:-ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa}"
PATCH_FILE="${PATCH_FILE:-$PACK_DIR/patches/qedispinv-python-interface.patch}"
UPDATE=0

usage() {
  cat <<'EOF'
Usage:
  ./install_qedispinv_to_conda.sh [options]

Install the patched QEDispInv Python modules into an existing conda env.

Options:
  --env NAME              Target conda environment. Default: ENV_NAME or adsurf_epvr.
  --src-root DIR          Source cache root. Default: SRC_ROOT or ~/.cache/adsurf-slope-normalized-loss/external_src.
  --qedispinv-ref REF     QEDispInv branch/tag/commit. Default: pinned commit documented in README.md.
  --qedispinv-url URL     QEDispInv git URL. Default: https://github.com/pan3rock/QEDispInv.git
  --update                Re-fetch and reset cached QEDispInv source before patching.
  -h, --help              Show this help.

Environment overrides:
  ENV_NAME, SRC_ROOT, QEDISPINV_URL, QEDISPINV_REF, PATCH_FILE
EOF
}

log() {
  printf '[qedispinv-install] %s\n' "$*"
}

die() {
  printf '[qedispinv-install] ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="${2:?missing value for --env}"
      shift 2
      ;;
    --src-root)
      SRC_ROOT="${2:?missing value for --src-root}"
      shift 2
      ;;
    --qedispinv-ref)
      QEDISPINV_REF="${2:?missing value for --qedispinv-ref}"
      shift 2
      ;;
    --qedispinv-url)
      QEDISPINV_URL="${2:?missing value for --qedispinv-url}"
      shift 2
      ;;
    --update)
      UPDATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

for cmd in conda git cmake make; do
  command -v "$cmd" >/dev/null 2>&1 || die "$cmd is not available in PATH"
done

[[ -f "$PATCH_FILE" ]] || die "missing patch file: $PATCH_FILE"

conda run -n "$ENV_NAME" python -c "import sys; print(sys.version)" >/dev/null \
  || die "conda env '$ENV_NAME' is not available"

conda run -n "$ENV_NAME" python -c "import pybind11; print(pybind11.get_cmake_dir())" >/dev/null \
  || die "pybind11 is missing in conda env '$ENV_NAME'. Install build deps first."

QEDISPINV_SRC_DIR="$SRC_ROOT/QEDispInv"
mkdir -p "$(dirname "$QEDISPINV_SRC_DIR")"

if [[ ! -d "$QEDISPINV_SRC_DIR/.git" ]]; then
  [[ ! -e "$QEDISPINV_SRC_DIR" ]] || die "$QEDISPINV_SRC_DIR exists but is not a git repository"
  log "cloning QEDispInv from $QEDISPINV_URL"
  git clone --recursive "$QEDISPINV_URL" "$QEDISPINV_SRC_DIR"
elif [[ "$UPDATE" == "1" ]]; then
  log "updating cached QEDispInv source"
  git -C "$QEDISPINV_SRC_DIR" reset --hard
  git -C "$QEDISPINV_SRC_DIR" clean -fd
  git -C "$QEDISPINV_SRC_DIR" fetch --tags origin
else
  log "using cached QEDispInv source: $QEDISPINV_SRC_DIR"
fi

log "checking out QEDispInv ref $QEDISPINV_REF"
git -C "$QEDISPINV_SRC_DIR" checkout "$QEDISPINV_REF"
git -C "$QEDISPINV_SRC_DIR" submodule update --init --recursive

if git -C "$QEDISPINV_SRC_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  log "applying the bundled QEDispInv Python-interface patch"
  git -C "$QEDISPINV_SRC_DIR" apply "$PATCH_FILE"
elif git -C "$QEDISPINV_SRC_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  log "patch is already applied"
else
  die "patch does not apply cleanly in $QEDISPINV_SRC_DIR"
fi

PYBIND11_DIR="$(conda run -n "$ENV_NAME" python -c "import pybind11; print(pybind11.get_cmake_dir())")"
CONDA_PREFIX="$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")"
SITE_PACKAGES="$(conda run -n "$ENV_NAME" python -c "import site; print(site.getsitepackages()[0])")"

log "building QEDispInv Python modules"
rm -rf "$QEDISPINV_SRC_DIR/build"
mkdir -p "$QEDISPINV_SRC_DIR/build"
cd "$QEDISPINV_SRC_DIR/build"

export PATH="${CONDA_PREFIX}/bin:${PATH}"
if command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1; then
  export CC="${CC:-x86_64-conda-linux-gnu-gcc}"
fi
if command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1; then
  export CXX="${CXX:-x86_64-conda-linux-gnu-g++}"
fi
if command -v x86_64-conda-linux-gnu-gfortran >/dev/null 2>&1; then
  export FC="${FC:-x86_64-conda-linux-gnu-gfortran}"
fi

CMAKE_PREFIX_PATH="$CONDA_PREFIX" cmake -Dpybind11_DIR="$PYBIND11_DIR" -DBUILD_PYTHON=ON ..
make -j

log "installing modules into $SITE_PACKAGES"
install -m 755 src/qedispinv_forward*.so "$SITE_PACKAGES/"
install -m 755 src/qedispinv_inversion*.so "$SITE_PACKAGES/"
if [[ -d "$QEDISPINV_SRC_DIR/python/qedispinv" ]]; then
  rm -rf "$SITE_PACKAGES/qedispinv"
  cp -a "$QEDISPINV_SRC_DIR/python/qedispinv" "$SITE_PACKAGES/"
fi

conda run -n "$ENV_NAME" python -c "import qedispinv, qedispinv_forward, qedispinv_inversion; print('QEDispInv import ok')"
log "done"
