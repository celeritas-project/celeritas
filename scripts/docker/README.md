# Docker images

These docker images use [spack](https://github.com/spack/spack) to build a
CUDA-enabled development environment for Celeritas. There are two sets of
images:
- `dev` (`dev` subdirectory) which leaves spack fully installed; and
- `ci` (`ci` subdirectory) which only copies the necessary software stack (thus
  requiring lower bandwidth on the CI servers).

Additionally there are two image configurations:
- `rocky-cuda12`: Rocky 9 with CUDA 12.
- `ubuntu-rocm6`: Ubuntu 24 with ROCm 6.3

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
$ docker run --rm -ti -e "TERM=xterm-256color" -v /rnsdhpc/code/celeritas-docker:/home/celeritas/src ci-ubuntu-rocm6
```

After mounting, use the build scripts to configure and go:
```console
celeritas@abcd1234:~$ cd src
celeritas@abcd1234:~/src$ ./scripts/docker/ci/run-ci.sh valgrind
```

Note that running as the `root` user requires the `MPIEXEC_PREFLAGS=--allow-run-as-root` to be defined for CMake: this is done by cmake-presets/ci-rocky-cuda.

