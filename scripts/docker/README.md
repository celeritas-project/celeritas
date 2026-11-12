# Docker images

## GPU images

The `gpu` subdirectory builds CUDA and ROCm images for the `build-docker` and
`build-codecov` workflows. They install Celeritas dependencies with
[Spack](https://github.com/spack/spack), using the Spack and spack-packages
commits pinned in `.github/actions/setup-spack/action.yml` and the shared
configuration in `scripts/spack`. The multi-stage `gpu/Dockerfile` has two targets:
- `dev`, which keeps Spack and the environment in `/opt/spack-environment`; and
- `ci`, which only copies the installed software (`/opt/software`,
  `/opt/view`) onto the base image, so CI downloads less.

There are two image configurations, both based on Ubuntu 24.04:
- `cuda`: CUDA 13.5 (`sm_80`), including VecGeom 2 with CUDA.
- `rocm`: ROCm 7.2 (`gfx90a`), without VecGeom.

The Spack environments are `scripts/spack/env-docker-<config>.yaml`, which
share `reqs-docker.yaml` (install tree, view, and system GCC). The CUDA or ROCm
toolkit and libraries of the base image are added as externals by
`gpu/write-gpu-externals.sh`, so their version is set only by the base image
tag in `build.sh`. Keep the GPU architectures in the environments consistent
with `scripts/cmake-presets/ci-ubuntu-{cuda,rocm}.json`.

### Building

Run `build.sh cuda` or `build.sh rocm`. It builds the `dev` and `ci` targets,
tags them `ghcr.io/celeritas-project/{dev,ci}-<config>:<date>` (set `REGISTRY`
to change the prefix), and prints the command to push the CI image (set
`PUSH=1` to push it directly). After pushing, update the
image tags in `.github/workflows/build-docker.yml` and `build-codecov.yml`.
Like the buildcache image, the build uses the container engine in
`_container.sh` and needs an `x86_64_v3` (AVX2) Linux host: CPU emulation is
far too slow to build these dependencies.

Pushing requires a GitHub personal access token (classic) with the
`write:packages` scope, authorized for the `celeritas-project` organization:
```console
$ docker login ghcr.io -u <username>
```
A newly created package is private: make it public in its GitHub package
settings so that CI can pull it.

After changing the Spack pins in the setup-spack action, rebuild the images so
that the GPU and CPU CI use the same package recipes.

### Running

To run the CI image with your local Celeritas source directory mounted:
```console
$ docker run --rm -ti -e "TERM=xterm-256color" \
    -v ${SOURCE}:/src -w /src \
    ghcr.io/celeritas-project/ci-cuda:${DATE}
```
where `${SOURCE}` is your local Celeritas source directory and `${DATE}` is the
date stamp of the desired image (or use the local tag `ci-cuda` if you
just built it). The `--rm` option deletes the container state, including any
build, when you exit. Inside the container, configure with the CI presets:
```console
# cp scripts/cmake-presets/ci-ubuntu-cuda.json CMakeUserPresets.json
# cmake --preset=release-vecgeom && cmake --build build
# cd build && CELER_DISABLE_DEVICE=1 ctest
```
(`CELER_DISABLE_DEVICE` is only needed on machines without a GPU.)
Running as `root` requires `MPIEXEC_PREFLAGS=--allow-run-as-root` for
CMake: the CI presets define it.

## Spack buildcache

Pull requests install their Spack dependencies only from the
`ghcr.io/celeritas-project/spack-buildcache` binary cache, so new dependency
combinations in the CI matrix must be pushed there first by
`scripts/ci/update-spack-buildcache.sh`. The `buildcache` image
reproduces the toolchain of the GitHub `ubuntu-24.04` runners (the externals
in `scripts/spack/ext-ubuntu24.yaml`) so that script can run on any x86_64
Linux host. Build it with:
```console
$ scripts/docker/buildcache/build.sh
```
The script reads the Spack and spack-packages commits pinned in
`.github/actions/setup-spack/action.yml` and passes them as build arguments.
It tags two images with the abbreviated commits:
`celeritas/spack-ubuntu24:<spack>-<packages>`, the runner toolchain with Spack
checked out and bootstrapped, and `celeritas-buildcache:<spack>-<packages>`
(also tagged `latest`), which runs the update script. Extra arguments are
passed to each build command.

Docker builds use BuildKit (`docker buildx build`), so the
[buildx plugin](https://docs.docker.com/go/buildx/) must be installed: the
legacy builder is deprecated. If Docker is missing, lacks buildx, or cannot
reach its daemon (e.g. without root or `docker` group access), the script
falls back to `podman`. Set `CONTAINER` to choose the container engine
explicitly (e.g. `CONTAINER=podman-hpc`).

Pushing requires a GitHub personal access token (classic) with the
`write:packages`, and `delete:packages` scopes, authorized for the `celeritas-project` organization.
Put the credentials in a private file such as `buildcache.env`:
```sh
GITHUB_USER=your-github-username
GITHUB_TOKEN=ghp_...
```
and run the update script with your Celeritas checkout mounted at `/celeritas`:
```console
$ docker run --rm -it \
    -v "$PWD:/celeritas:ro" \
    -v celeritas-opt-ci:/scratch/celeritas/opt-ci \
    --env-file buildcache.env \
    celeritas-buildcache
```
Append `bash` to the command for an interactive shell instead.

With rootless podman, such as `podman-hpc` on Perlmutter, the container's
`celeritas` user cannot read files owned by your host user: run as container
root (which is your own user) and keep the installations in a scratch directory
so they are visible from every node:
```console
$ CONTAINER=podman-hpc scripts/docker/buildcache/build.sh
$ mkdir -p $SCRATCH/celeritas-opt-ci
$ podman-hpc run --rm -it --user 0 \
    -v "$PWD:/celeritas:ro" \
    -v $SCRATCH/celeritas-opt-ci:/scratch/celeritas/opt-ci \
    --env-file buildcache.env \
    celeritas-buildcache
```

Notes:
- The entrypoint refuses to run if the Spack commits in the image differ from
  the ones pinned in the mounted `.github/actions/setup-spack/action.yml`, so
  the concretization always matches CI. After updating the pins, rerun
  `build.sh`: only the Spack layers are rebuilt.
- The host CPU must support `x86_64_v3` (AVX2), the target required by
  `scripts/spack/reqs-ci.yaml`: CPU emulation is not supported.
- The `celeritas-opt-ci` volume keeps the installed packages,
  so rerunning after a failure or a matrix change only builds what is missing.
- Do not mount a volume at `/work`: the update script skips environment
  directories that already exist there, which would silently skip
  environments after the Spack version or the matrix changes.
- Build logs of failed packages (including `config.log`) stay in the stage
  directory under `/tmp/<user>/spack-stage` inside the container. The update
  script stops at the first failure, so omit `--rm` to keep the container and
  copy them out with `docker cp`.
