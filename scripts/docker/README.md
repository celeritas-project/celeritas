# Docker images

These docker images use [spack](https://github.com/spack/spack) to build a
CUDA-enabled development environment for Celeritas. There are two sets of
images:
- `dev` (`dev` subdirectory) which leaves spack fully installed and
  debug symbols intact; and
- `ci` (`ci` subdirectory) which only copies the necessary software stack (thus
  requiring lower bandwidth on the CI servers).

Additionally there are two image configurations:
- `focal-cuda11`: Ubuntu 20 "Focal" with CUDA 11.
- `bionic-minimal`: Ubuntu 18 "Bionic" with only googletest/nljson.

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

The `execute-local-test` script will clone an active GitHub pull request, build,
and set up an image to use locally:
```console
$ ./ci/execute-local-test.sh 123
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
$ docker run --rm -ti -e "TERM=xterm-256color" -v /rnsdhpc/code/celeritas-docker:/home/celeritas/src ci-bionic-minimal
```

After mounting, use the build scripts to configure and go:
```console
celeritas@abcd1234:~$ cd src
celeritas@abcd1234:~/src$ ./scripts/docker/ci/run-ci.sh valgrind
```

The `dev` image runs as root, but the `ci-focal-cuda11` runs as a user
`celeritas`.  This is the best way to [make OpenMPI
happy](https://github.com/open-mpi/ompi/issues/4451).

Note that the Jenkins CI runs as root regardless of the `run` command, so it
defines `MPIEXEC_PREFLAGS=--allow-run-as-root` for CMake.

