#!/usr/bin/env bash
# Build AWS Palace from source into a conda-forge toolchain environment.
#   PREFIX  : environment created from environment.yml (Palace installs into it)
#   BUILD   : scratch directory (use a large, non-home disk; ~6 GB during build)
#   JOBS    : parallel make jobs (keep low on a shared pod; the dependency
#             superbuild peaks at ~1.5 GB RSS per job)
set -euo pipefail
PALACE_VERSION=${PALACE_VERSION:-v0.18.1}
PREFIX=${PREFIX:-/tmp/ose-envs/palace}
BUILD=${BUILD:-/tmp/ose-build}
JOBS=${JOBS:-3}

export PATH="$PREFIX/bin:$PATH"
export OPENBLAS_DIR="$PREFIX"
export CC=mpicc CXX=mpicxx FC=mpifort
export OMPI_CC="$PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export OMPI_CXX="$PREFIX/bin/x86_64-conda-linux-gnu-g++"
export OMPI_FC="$PREFIX/bin/x86_64-conda-linux-gnu-gfortran"

mkdir -p "$BUILD"
[ -d "$BUILD/palace" ] || git clone --depth 1 --branch "$PALACE_VERSION" https://github.com/awslabs/palace.git "$BUILD/palace"
mkdir -p "$BUILD/palace/build" && cd "$BUILD/palace/build"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DBUILD_SHARED_LIBS=OFF \
  -DPALACE_WITH_OPENMP=OFF \
  -DPALACE_WITH_SUPERLU=ON \
  -DPALACE_WITH_SLEPC=ON \
  -DPALACE_WITH_ARPACK=OFF \
  -DPALACE_WITH_MAGMA=OFF \
  -DPALACE_WITH_LIBXSMM=OFF \
  -DPALACE_WITH_SUNDIALS=OFF \
  -DPALACE_WITH_GSLIB=ON

# conda-forge make 4.4 defaults to a fifo jobserver; gslib's sub-build calls the
# system gmake 4.3, which cannot read it ("invalid --jobserver-auth"). Force pipes.
make -j "$JOBS" --jobserver-style=pipe
"$PREFIX/bin/palace" --help | head -5 || true
