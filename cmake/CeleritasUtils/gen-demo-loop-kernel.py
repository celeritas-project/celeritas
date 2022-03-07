#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Tool to generate demo loop kernel implementations on the fly.
"""

import os.path
import sys
from launchbounds import make_launch_bounds

CLIKE_TOP = '''\
//{modeline:-^75s}//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//! \\note Auto-generated by {script}: DO NOT MODIFY!
//---------------------------------------------------------------------------//
'''

HH_TEMPLATE = CLIKE_TOP + """\
#include "base/Assert.hh"
#include "base/Macros.hh"

namespace demo_loop
{{
namespace generated
{{
void {func}(
    const celeritas::ParamsHostRef&,
    const celeritas::StateHostRef&);

void {func}(
    const celeritas::ParamsDeviceRef&,
    const celeritas::StateDeviceRef&);

#if !CELER_USE_DEVICE
inline void {func}(
    const celeritas::ParamsDeviceRef&,
    const celeritas::StateDeviceRef&)
{{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}}
#endif

}} // namespace generated
}} // namespace demo_loop
"""

CC_TEMPLATE = CLIKE_TOP + """\
#include "base/Assert.hh"
#include "base/Types.hh"
#include "../LDemoLauncher.hh"

using namespace celeritas;

namespace demo_loop
{{
namespace generated
{{
void {func}(
    const ParamsHostRef& params,
    const StateHostRef& states)
{{
    CELER_EXPECT(params);
    CELER_EXPECT(states);

    {class}Launcher<MemSpace::host> launch(params, states);
    #pragma omp parallel for
    for (size_type i = 0; i < {threads}; ++i)
    {{
        launch(ThreadId{{i}});
    }}
}}

}} // namespace generated
}} // namespace demo_loop
"""

CU_TEMPLATE = CLIKE_TOP + """\
#include "base/device_runtime_api.h"
#include "base/Assert.hh"
#include "base/Types.hh"
#include "base/KernelParamCalculator.device.hh"
#include "comm/Device.hh"
#include "../LDemoLauncher.hh"

using namespace celeritas;

namespace demo_loop
{{
namespace generated
{{
namespace
{{
__global__ void{launch_bounds}{func}_kernel(
    ParamsDeviceRef const params,
    StateDeviceRef const states)
{{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < {threads}))
        return;

    {class}Launcher<MemSpace::device> launch(params, states);
    launch(tid);
}}
}} // namespace

void {func}(
    const celeritas::ParamsDeviceRef& params,
    const celeritas::StateDeviceRef& states)
{{
    CELER_EXPECT(params);
    CELER_EXPECT(states);
    CELER_LAUNCH_KERNEL({func},
                        celeritas::device().default_block_size(),
                        {threads},
                        params, states);
}}

}} // namespace generated
}} // namespace demo_loop
"""

TEMPLATES = {
    'hh': HH_TEMPLATE,
    'cc': CC_TEMPLATE,
    'cu': CU_TEMPLATE,
}
LANG = {
    'hh': "C++",
    'cc': "C++",
    'cu': "CUDA",
}

def generate(**subs):
    ext = subs['ext']
    subs['modeline'] = "-*-{}-*-".format(LANG[ext])
    template = TEMPLATES[ext]
    filename = "{basename}.{ext}".format(**subs)
    subs['filename'] = filename
    subs['script'] = os.path.basename(sys.argv[0])
    subs['launch_bounds'] = make_launch_bounds(subs['func'])
    with open(filename, 'w') as f:
        f.write(template.format(**subs))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--basename',
        help='File name (without extension) of output')
    parser.add_argument(
        '--class',
        help='CamelCase name of the class prefix')
    parser.add_argument(
        '--func',
        help='snake_case name of the function')
    parser.add_argument(
        '--threads',
        default='states.size()',
        help='String describing the number of threads')

    kwargs = vars(parser.parse_args())
    for ext in ['hh', 'cc', 'cu']:
        generate(ext=ext, **kwargs)

if __name__ == '__main__':
    main()
