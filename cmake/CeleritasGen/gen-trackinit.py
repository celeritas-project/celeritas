#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Tool to generate celeritas/track/InitTrackUtils implementations on the fly.
"""

import sys
from pathlib import Path
from collections import namedtuple
from launchbounds import make_launch_bounds

CLIKE_TOP = '''\
//{modeline:-^75s}//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//! \\note Auto-generated by {script}: DO NOT MODIFY!
//---------------------------------------------------------------------------//
'''

HH_TEMPLATE = """\
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"
{extra_includes}

namespace celeritas
{{
namespace generated
{{

{hostfunc_decl};

{devicefunc_decl};

#if !CELER_USE_DEVICE
inline {devicefunc_decl_noargs}
{{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}}
#endif

}}  // namespace generated
}}  // namespace celeritas
"""

CC_TEMPLATE = CLIKE_TOP + """\
#include <utility>

#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "corecel/Types.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/track/detail/{clsname}Launcher.hh" // IWYU pragma: associated

namespace celeritas
{{
namespace generated
{{
{hostfunc_decl}
{{
    MultiExceptionHandler capture_exception;
    detail::{clsname}Launcher<MemSpace::host> launch({kernel_arglist});
    #pragma omp parallel for
    for (ThreadId::size_type i = 0; i < {num_threads}; ++i)
    {{
        CELER_TRY_HANDLE_CONTEXT(
            launch(ThreadId{{i}}),
            capture_exception,
            KernelContextException(core_params, core_states, ThreadId{{i}}, "{funcname}"));
    }}
    log_and_rethrow(std::move(capture_exception));
}}

}}  // namespace generated
}}  // namespace celeritas
"""

CU_TEMPLATE = CLIKE_TOP + """\
#include "celeritas/track/detail/{clsname}Launcher.hh"
#include "corecel/device_runtime_api.h"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/Types.hh"

namespace celeritas
{{
namespace generated
{{
namespace
{{
{kernel_decl}
{{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < {num_threads}))
        return;

    detail::{clsname}Launcher<MemSpace::device> launch({kernel_arglist});
    launch(tid);
}}
}}  // namespace

{devicefunc_decl}
{{
    CELER_LAUNCH_KERNEL(
        {funcname},
        celeritas::device().default_block_size(),
        {num_threads},
        {kernel_arglist});
}}

}}  // namespace generated
}}  // namespace celeritas
"""

Param = namedtuple("Param", ["type", "name"])


def make_const(type_name):
    if not type_name.endswith(' const'):
        return type_name + ' const'
    return type_name


def make_const_reference(type_name):
    if type_name.startswith('Span<') or type_name == 'size_type':
        return make_const(type_name)
    return type_name + ' const&'


class ParamList(list):
    @property
    def decl(self):
        return ",".join("\n    " + " ".join(p) for p in self)

    @property
    def decl_noargs(self):
        return ", ".join(p.type for p in self)

    @property
    def arglist(self):
        return ", ".join(p.name for p in self)


class Function(namedtuple("Function", ["name", "params"])):
    __slots__ = ()

    @property
    def decl(self):
        result = ["void ", self.name, "(", self.params.decl, ")"]
        return "".join(result)

    @property
    def decl_noargs(self):
        result = ["void ", self.name, "(", self.params.decl_noargs, ")"]
        return "".join(result)


def make_kernel_decl(func):
    # All args are const
    params = ParamList([p._replace(type=make_const(p.type))
                        for p in func.params])
    # Kernel name gets a suffix
    name = func.name + "_kernel"
    # Generate launch_bounds
    launch_bounds = make_launch_bounds(func.name)

    result = ["__global__ void", launch_bounds, name,
              "(", params.decl, ")"]
    return "".join(result)


KernelDefinition = namedtuple('KernelDefinition', [
"function",
"num_threads",
"includes"])

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

DEFS = {
    "InitTracks": KernelDefinition(
        Function("init_tracks", ParamList([
            Param("{Memspace}CRef<CoreParamsData>", "core_params"),
            Param("{Memspace}Ref<CoreStateData>", "core_states"),
            Param("size_type", "num_vacancies"),
        ])),
        "num_vacancies",
        []),
    "LocateAlive": KernelDefinition(
        Function("locate_alive", ParamList([
            Param("{Memspace}CRef<CoreParamsData>", "core_params"),
            Param("{Memspace}Ref<CoreStateData>", "core_states"),
        ])),
        "core_states.size()",
        []),
    "ProcessPrimaries": KernelDefinition(
        Function("process_primaries", ParamList([
            Param("{Memspace}CRef<CoreParamsData>", "core_params"),
            Param("{Memspace}Ref<CoreStateData>", "core_states"),
            Param("Span<const Primary>", "primaries"),
        ])),
        "primaries.size()",
        ["corecel/cont/Span.hh", "celeritas/phys/Primary.hh"]),
    "ProcessSecondaries": KernelDefinition(
        Function("process_secondaries", ParamList([
            Param("{Memspace}CRef<CoreParamsData>", "core_params"),
            Param("{Memspace}Ref<CoreStateData>", "core_states"),
        ])),
        "core_states.size()",
        []),
}

def transformed_param_types(params, apply, **kwargs):
    return ParamList([p._replace(type=apply(p.type.format(**kwargs)))
                      for p in params])

def generate(*, ext, **kwargs):
    template = TEMPLATES[ext]
    lang = LANG[ext]

    kwargs['modeline'] = "-*-{}-*-".format(lang)
    script = Path(sys.argv[0])
    filename = Path("{basename}.{ext}".format(ext=ext, **kwargs))
    kwargs['filename'] = Path(kwargs['basedir']) / filename
    kwargs['script'] = script.name

    kdef = DEFS[kwargs['clsname']]
    kwargs['extra_includes'] = "\n".join("#include \"{}\"".format(fn)
                                         for fn in kdef.includes)

    (funcname, params) = kdef.function
    kwargs['funcname'] = funcname
    kwargs['num_threads'] = kdef.num_threads
    kwargs['memspace'] = "host"
    kwargs['Memspace'] = "Host"
    host_params = transformed_param_types(params, make_const_reference, **kwargs)
    kwargs['memspace'] = "device"
    kwargs['Memspace'] = "Device"
    device_params = transformed_param_types(params, make_const_reference, **kwargs)
    kernel_params = transformed_param_types(params, make_const, **kwargs)

    host_func = Function(funcname, host_params)
    device_func = Function(funcname, device_params)

    kwargs['hostfunc_decl'] = host_func.decl
    kwargs['devicefunc_decl'] = device_func.decl
    kwargs['devicefunc_decl_noargs'] = device_func.decl_noargs
    kwargs['kernel_decl'] = make_kernel_decl(Function(funcname, kernel_params))
    kwargs['kernel_arglist'] = kernel_params.arglist


    with open(filename, 'w') as f:
        f.write(template.format(**kwargs))

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
        '--class', dest='clsname',
        help='CamelCase name of the class prefix')

    kwargs = vars(parser.parse_args())
    for ext in ['hh', 'cc', 'cu']:
        generate(ext=ext, **kwargs)

if __name__ == '__main__':
    main()
