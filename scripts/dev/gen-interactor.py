#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Tool to generate Model "interact" implementations on the fly.

Assumptions:
 - Generated in a subdirectory (preferably named ``generated``) next to the
   ``detail/Class.hh`` directory
 - The class's header defines ClassDeviceRef and ClassDeviceHost type aliases
   for the host/device collection groups.
"""

FILENAME = "{class}Interact.{ext}"

CLIKE_TOP = '''\
//{modeline:-^75s}//
// Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//! \\note Auto-generated by gen-interactor.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
'''

HH_TEMPLATE = CLIKE_TOP + """\
#include "celeritas_config.h"
#include "base/Assert.hh"
#include "../detail/{class}.hh"

namespace celeritas
{{
namespace generated
{{
void {func}_interact(
    const detail::{class}HostRef&,
    const ModelInteractRefs<MemSpace::host>&);

void {func}_interact(
    const detail::{class}DeviceRef&,
    const ModelInteractRefs<MemSpace::device>&);

#if !CELERITAS_USE_CUDA
inline void {func}_interact(
    const detail::{class}DeviceRef&,
    const ModelInteractRefs<MemSpace::device>&)
{{
    CELER_ASSERT_UNREACHABLE();
}}
#endif

}} // namespace generated
}} // namespace celeritas
"""

CC_TEMPLATE = CLIKE_TOP + """\
#include "base/Assert.hh"
#include "base/Range.hh"
#include "base/Types.hh"
#include "../detail/{class}.hh"

namespace celeritas
{{
namespace generated
{{
void {func}_interact(
    const detail::{class}HostRef& ptrs,
    const ModelInteractRefs<MemSpace::host>& model)
{{
    CELER_EXPECT(ptrs);
    CELER_EXPECT(model);

    detail::{class}Launcher<MemSpace::host> launch(ptrs, model);
    for (auto tid : range(ThreadId{{model.states.size()}}))
    {{
        launch(tid);
    }}
}}

}} // namespace generated
}} // namespace celeritas
"""

CU_TEMPLATE = CLIKE_TOP + """\
#include "base/Assert.hh"
#include "base/KernelParamCalculator.cuda.hh"
#include "../detail/{class}.hh"

using namespace celeritas::detail;

namespace celeritas
{{
namespace generated
{{
namespace
{{
__global__ void {func}_interact_kernel(
    const detail::{class}DeviceRef ptrs,
    const ModelInteractRefs<MemSpace::device> model)
{{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < model.states.size()))
        return;

    detail::{class}Launcher<MemSpace::device> launch(ptrs, model);
    launch(tid);
}}
}} // namespace

void {func}_interact(
    const detail::{class}DeviceRef& ptrs,
    const ModelInteractRefs<MemSpace::device>& model)
{{
    CELER_EXPECT(ptrs);
    CELER_EXPECT(model);

    static const KernelParamCalculator calc_kernel_params(
        {func}_interact_kernel, "{func}_interact");
    auto params = calc_kernel_params(model.states.size());
    {func}_interact_kernel<<<params.grid_size, params.block_size>>>(
        ptrs, model);
    CELER_CUDA_CHECK_ERROR();
}}

}} // namespace generated
}} // namespace celeritas
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
    subs['modeline'] = "-*-{}-*-".format(ext)
    template = TEMPLATES[ext]
    filename = FILENAME.format(**subs)
    subs['filename'] = filename
    with open(filename, 'w') as f:
        f.write(template.format(**subs))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--class',
        help='CamelCase name of the class prefix')
    parser.add_argument(
        '--func',
        help='snake_case name of the interact function prefix')

    kwargs = vars(parser.parse_args())
    kwargs['year'] = 2021
    for ext in ['hh', 'cc', 'cu']:
        generate(ext=ext, **kwargs)

if __name__ == '__main__':
    main()
