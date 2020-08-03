# Docker images

These docker images use [spack](https://github.com/spack/spack) to build a
CUDA-enabled development environment for Celeritas. There are two images:
- `celeritas:dev` (this directory) which leaves spack fully installed and
  debug symbols intact; and
- `celeritas:ci-cuda11` (`ci` subdirectory) which only copies the
  necessary software stack.

Both images are based on `nvidia` Ubuntu installations.

## Building

From inside this directory, you can build and tag both versions:

```console
$ docker build -t celeritas/dev:prereq .
$ docker build -t celeritas/ci:cuda11 ci
```

## Running

The development image is (in color) run with:
```console
$ docker run --rm -ti -e "TERM=xterm-256color" celeritas:dev
```
Note that the `--rm` option automatically deletes the state of the container
after you exit the docker client. This means all of your work will be
destroyed.

The `dev` image runs as root, but the `ci-cuda11` runs as a user "celeritas".
This is the best way to [make OpenMPI happy](https://github.com/open-mpi/ompi/issues/4451).

## Pushing

The CI docker images should be pushed upstream to enable continuous
integration. This is a one-line command:
```console
$ docker push celeritas/ci:cuda11
```
but it requires that you log in with your `docker.io` credentials first:
```console
$ docker login -u sethrj
```
