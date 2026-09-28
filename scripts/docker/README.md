# Docker images

These docker images use [spack](https://github.com/spack/spack) to build a
CUDA-enabled development environment for Celeritas. There are two sets of
images:
- `dev` (`dev` subdirectory) which leaves spack fully installed; and
- `ci` (`ci` subdirectory) which only copies the necessary software stack (thus
  requiring lower bandwidth on the CI servers).

Additionally there are two image configurations:
- `rocky-cuda12`: Rocky 9 with CUDA 12.
- `ubuntu-rocm7`: Ubuntu 24 with ROCm 7.1.

## Building

The included `build.sh` script drives the two subsequent docker builds. Its
argument should be an image configuration name, or `cuda` or `minimal` as
shortcuts for those.

If the final docker push fails, you may have to log in with your `docker.io`
credentials first:
```console
$ docker login -u sethrj
```

## Running

The CI image is (in color) run with:
```console
$ docker run --rm -ti -e "TERM=xterm-256color" celeritas/ci-cuda11
```
Note that the `--rm` option automatically deletes the state of the container
after you exit the docker client. This means all of your work will be
destroyed.

The `launch-local-test` script will clone an active GitHub pull request, build,
and set up an image to use locally:
```console
$ ./ci/launch-local-test.sh 123
```

To mount the image with your local source directory:
```console
$ docker run --rm -ti -e "TERM=xterm-256color" \
    -v ${SOURCE}:/home/celeritas/src \
    celeritas/ci-focal-cuda11:${DATE}
```
where `${SOURCE}` is your local Celeritas source directory and `${DATE}` is the
date time stamp of the desired image. If you just built locally, you can
replace that last argument with the tag `ci-focal-cuda11`:
```console
$ docker run --rm -ti -e "TERM=xterm-256color" -v /rnsdhpc/code/celeritas-docker:/home/celeritas/src ci-ubuntu-rocm7
```

After mounting, use the build scripts to configure and go:
```console
celeritas@abcd1234:~$ cd src
celeritas@abcd1234:~/src$ ./scripts/docker/ci/run-ci.sh valgrind
```

Note that running as the `root` user requires the `MPIEXEC_PREFLAGS=--allow-run-as-root` to be defined for CMake: this is done by cmake-presets/ci-rocky-cuda.

## Spack buildcache

Pull requests install their Spack dependencies only from the
`ghcr.io/celeritas-project/spack-buildcache` binary cache, so new dependency
combinations in the CI matrix must be pushed there first by
`scripts/ci/update-spack-buildcache-local.sh`. The `buildcache` image
reproduces the toolchain of the GitHub `ubuntu-24.04` runners (the externals
in `scripts/spack/ext-ubuntu24.yaml`) so that script can run on any x86_64
Linux host. Build it from the top-level source directory:
```console
$ docker build -f scripts/docker/buildcache/Dockerfile -t celeritas-buildcache .
```

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
$ mkdir -p $SCRATCH/celeritas-opt-ci
$ podman-hpc run --rm -it --user 0 \
    -v "$PWD:/celeritas:ro" \
    -v $SCRATCH/celeritas-opt-ci:/scratch/celeritas/opt-ci \
    --env-file buildcache.env \
    celeritas-buildcache
```

Notes:
- The entrypoint checks out the Spack and spack-packages commits pinned in the
  mounted `.github/actions/setup-spack/action.yml`, so the concretization
  matches CI even if the image is older than the pins.
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
