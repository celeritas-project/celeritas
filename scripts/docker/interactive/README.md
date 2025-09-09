# Interactive container

Interactive CPU only container with Celeritas and its dependencies compiled with Clang.
Requires root privilege to use Linux ftrace

## Build
Install all necessary Celeritas dependencies, checkout and compile a Celeritas.
The container tag matches the git branch/tag name:
```shell
# -t git branch or tag to checkout, default develop
# -r git repo url, default celeritas-project/celeritas
./build.sh
```

## Run
By default, mount a volume in ```/data``` that persist data across
executions and bind the celeritas root directory to ```/host```
```shell
# -t container tag to run, default develop
# -m host directory to mount inside the container under /host, default celeritas root dir
./run.sh
```
