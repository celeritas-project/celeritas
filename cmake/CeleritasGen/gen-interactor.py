#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Tool to generate Model "interact" implementations on the fly.

Assumptions:
 - The class's header defines ClassDeviceRef and ClassDeviceHost type aliases
   for the host/device collection groups.
"""

import sys
from pathlib import Path
from launchbounds import make_launch_bounds

CLIKE_TOP = '''\
//{modeline:-^75s}//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//! \\note Auto-generated by {script}: DO NOT MODIFY!
//---------------------------------------------------------------------------//
'''

HH_TEMPLATE = CLIKE_TOP + """\
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/{dir}/data/{class}Data.hh" // IWYU pragma: associated
#include "celeritas/global/CoreTrackData.hh"

namespace {namespace}
{{
namespace generated
{{
void {func}_interact(
    {namespace}::{class}HostRef const&,
    celeritas::CoreRef<celeritas::MemSpace::host> const&);

void {func}_interact(
    {namespace}::{class}DeviceRef const&,
    celeritas::CoreRef<celeritas::MemSpace::device> const&);

#if !CELER_USE_DEVICE
inline void {func}_interact(
    {namespace}::{class}DeviceRef const&,
    celeritas::CoreRef<celeritas::MemSpace::device> const&)
{{
    CELER_ASSERT_UNREACHABLE();
}}
#endif

}}  // namespace generated
}}  // namespace {namespace}
"""

CC_TEMPLATE = CLIKE_TOP + """\
#include "{class}Interact.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/{dir}/launcher/{class}Launcher.hh" // IWYU pragma: associated
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace {namespace}
{{
namespace generated
{{
void {func}_interact(
    {namespace}::{class}HostRef const& model_data,
    celeritas::CoreRef<MemSpace::host> const& core_data)
{{
    CELER_EXPECT(core_data);
    CELER_EXPECT(model_data);

    celeritas::MultiExceptionHandler capture_exception;
    auto launch = celeritas::make_interaction_launcher(
        core_data,
        model_data,
        {namespace}::{func}_interact_track);
    #pragma omp parallel for
    for (celeritas::size_type i = 0; i < core_data.states.size(); ++i)
    {{
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{{i}}),
            capture_exception,
            KernelContextException(core_data, ThreadId{{i}}, "{func}"));
    }}
    log_and_rethrow(std::move(capture_exception));
}}

}}  // namespace generated
}}  // namespace {namespace}
"""

CU_TEMPLATE = CLIKE_TOP + """\
#include "{class}Interact.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/{dir}/launcher/{class}Launcher.hh"
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace {namespace}
{{
namespace generated
{{
namespace
{{
__global__ void{launch_bounds}{func}_interact_kernel(
    {namespace}::{class}DeviceRef const model_data,
    celeritas::CoreRef<MemSpace::device> const core_data)
{{
    auto tid = celeritas::KernelParamCalculator::thread_id();
    if (!(tid < core_data.states.size()))
        return;

    auto launch = celeritas::make_interaction_launcher(
        core_data,
        model_data,
        {namespace}::{func}_interact_track);
    launch(tid);
}}
}}  // namespace

void {func}_interact(
    {namespace}::{class}DeviceRef const& model_data,
    celeritas::CoreRef<MemSpace::device> const& core_data)
{{
    CELER_EXPECT(core_data);
    CELER_EXPECT(model_data);

    CELER_LAUNCH_KERNEL({func}_interact,
                        celeritas::device().default_block_size(),
                        core_data.states.size(),
                        model_data, core_data);
}}

}}  // namespace generated
}}  // namespace {namespace}
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
    script = Path(sys.argv[0])
    filename = Path("{basename}.{ext}".format(**subs))
    subs['filename'] = Path(subs['basedir']) / filename
    subs['script'] = script.name
    subs['launch_bounds'] = make_launch_bounds(subs['func'] + '_interact')
    with open(filename, 'w') as f:
        f.write(template.format(**subs))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--basedir',
        default='celeritas',
        help='execution path relative to src/')
    parser.add_argument(
        '--basename',
        help='File name (without extension) of output')
    parser.add_argument(
        '--class',
        help='CamelCase name of the class prefix')
    parser.add_argument(
        '--func',
        help='snake_case name of the interact function prefix')
    parser.add_argument(
        '--dir',
        default='em',
        help='directory inside celeritas')
    parser.add_argument(
        '--namespace',
        default='celeritas',
        help='namespace of model/process/launcher/etc')


    kwargs = vars(parser.parse_args())
    for ext in ['hh', 'cc', 'cu']:
        generate(ext=ext, **kwargs)

if __name__ == '__main__':
    main()
