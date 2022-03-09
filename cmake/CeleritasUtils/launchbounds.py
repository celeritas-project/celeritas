# -*- coding: utf-8 -*-
# Copyright 2022 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""

"""

def load_bounds():
    import json
    from pathlib import Path

    with open(Path(__file__).parent / 'launch-bounds.json') as f:
        result = json.load(f)

    return (result['devices'], result['kernels'])

def build_cuda_opts(device, opt):
    cuda_arch = (device['capability_major'] * 100
                 + device['capability_minor'] * 10)
    return f"""\
#if CELERITAS_USE_CUDA && (__CUDA_ARCH__ == {cuda_arch}) // {device['name']}
__launch_bounds__({opt['max_threads_per_block']}, {opt['min_blocks_per_cu']})
#endif"""


def build_hip_opts(device, opt):
    return f"""\
#if CELERITAS_USE_HIP && defined(__{device['name']}__)
__launch_bounds__({opt['max_threads_per_block']}, {opt['min_warps_per_eu']})
#endif"""


BUILD_OPTS = {
    'cuda': build_cuda_opts,
    'hip': build_hip_opts,
}

def make_launch_bounds(kernel_name):
    (devices, kernels) = load_bounds()
    try:
        opts = kernels[kernel_name]
    except KeyError:
        # Kernel doesn't exist in the list of precalculated bounds
        return " "

    assert len(devices) == len(opts)

    result = ["\n#if CELERITAS_LAUNCH_BOUNDS"]
    for device, opt in zip(devices, opts):
        build_opts = BUILD_OPTS[device['platform']]
        result.append(build_opts(device, opt))

    result.append("#endif // CELERITAS_LAUNCH_BOUNDS\n")
    return "\n".join(result)

