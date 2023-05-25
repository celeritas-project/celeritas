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

namespace celeritas
{{
class CoreParams;
template<MemSpace M>
class CoreState;
}}

namespace {namespace}
{{
namespace generated
{{
void {func}_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::host>&,
    {namespace}::{class}HostRef const&);

void {func}_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&,
    {namespace}::{class}DeviceRef const&,
    celeritas::ActionId);

#if !CELER_USE_DEVICE
inline void {func}_interact(
    celeritas::CoreParams const&,
    celeritas::CoreState<MemSpace::device>&,
    {namespace}::{class}DeviceRef const&,
    celeritas::ActionId)
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
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/{dir}/launcher/{class}Launcher.hh" // IWYU pragma: associated
#include "celeritas/phys/InteractionLauncher.hh"

using celeritas::MemSpace;

namespace {namespace}
{{
namespace generated
{{
void {func}_interact(
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::host>& state,
    {namespace}::{class}HostRef const& model_data)
{{
    CELER_EXPECT(model_data);

    celeritas::MultiExceptionHandler capture_exception;
    auto launch = celeritas::make_interaction_launcher(
        params.ptr<MemSpace::native>(), state.ptr(),
        {namespace}::{func}_interact_track, model_data);
    #pragma omp parallel for
    for (celeritas::size_type i = 0; i < state.size(); ++i)
    {{
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{{i}}),
            capture_exception,
            KernelContextException(params.ref<MemSpace::host>(), state.ref(), ThreadId{{i}}, "{func}"));
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
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/KernelLaunchUtils.hh"
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
    celeritas::CRefPtr<celeritas::CoreParamsData, MemSpace::device> const params,
    celeritas::RefPtr<celeritas::CoreStateData, MemSpace::device> const state,
    {namespace}::{class}DeviceRef const model_data,
    celeritas::size_type size,
    celeritas::ThreadId const offset)
{{
    auto tid = celeritas::KernelParamCalculator::thread_id() + offset.get();
    if (!(tid < size))
        return;

    auto launch = celeritas::make_interaction_launcher(
        params, state, {namespace}::{func}_interact_track, model_data);
    launch(tid);
}}
}}  // namespace

void {func}_interact(
    celeritas::CoreParams const& params,
    celeritas::CoreState<MemSpace::device>& state,
    {namespace}::{class}DeviceRef const& model_data,
    celeritas::ActionId action)
{{
    CELER_EXPECT(model_data);
    KernelLaunchParams kernel_params = compute_launch_params(action, params, state, TrackOrder::sort_step_limit_action);
    CELER_LAUNCH_KERNEL({func}_interact,
                        celeritas::device().default_block_size(),
                        kernel_params.num_threads,
                        params.ptr<MemSpace::native>(),
                        state.ptr(),
                        model_data,
                        kernel_params.num_threads,
                        kernel_params.threads_offset);
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
