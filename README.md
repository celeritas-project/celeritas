# Celeritas

The Celeritas project implements HEP detector physics on GPU accelerator
hardware with the ultimate goal of supporting the massive computational
requirements of the [HL-LHC upgrade][HLLHC].

[HLLHC]: https://home.cern/science/accelerators/high-luminosity-lhc

# Installation

This project requires external dependencies such as CUDA to build with full
functionality.  However, any combination of these requirements can be omitted
to enable limited development on personal machines with fewer available
components. See [the infrastructure documentation](doc/infrastructure.rst) for
details on installing.

## Installing with Spack

[Spack](https://github.com/spack/spack) is an HPC-oriented package manager that
includes numerous scientific packages, including those used in HEP. An included
Spack "environment" (at `scripts/dev/env/celeritas-{platform}.yaml`) defines
the required prerequisites for this project.

- Clone Spack following its [getting started instructions][1]
- To install with CUDA, run `spack external find cuda` and
  `spack install celeritas +cuda cuda_arch=<ARCH>`, where `<ARCH>` is the
  numeric portion of the [CUDA architecture flags][2]

[1]: https://spack.readthedocs.io/en/latest/getting_started.html
[2]: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

## Configuring and building Celeritas manually

The Spack environment at [dev/scripts.yaml](dev/scripts.yaml) lists the full
dependencies used by the CI for building, testing, and documenting. Install
those dependencies via Spack or independently, then configure Celeritas.

To configure Celeritas, assuming the dependencies you want are located in the
`CMAKE_PREFIX_PATH` search path, and other environment variables such as `CXX`
are set, you should be able to just run CMake and build:
```console
$ mkdir build
$ cd build && cmake ..
$ make
```

# Development

See the [contribution guide](CONTRIBUTING.rst) for the contribution process,
[the development guidelines](doc/appendices/development.rst) for further
details on coding in Celeritas, and [the development guidelines](doc/appendices/development.rst).

# Documentation

The full code documentation (including API descriptions) is available by
setting the `CELERITAS_BUILD_DOCS=ON` configuration option. A mostly complete
version of the [Celeritas documentation][docs] is hosted on `readthedocs.io`.

[docs]: https://celeritas.readthedocs.io/en/latest/
