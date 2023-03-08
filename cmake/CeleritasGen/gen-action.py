#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Tool to generate simpled "Action"-based kernel implementations automatically.
"""

import sys
from pathlib import Path
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

HH_TEMPLATE = CLIKE_TOP + """\
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackData.hh"

namespace celeritas
{{
namespace generated
{{
//---------------------------------------------------------------------------//
class {clsname} final : public ExplicitActionInterface, public ConcreteAction
{{
public:
  // Construct with ID and label
  using ConcreteAction::ConcreteAction;

  // Launch kernel with host data
  void execute(CoreHostRef const&) const final;

  // Launch kernel with device data
  void execute(CoreDeviceRef const&) const final;

  //! Dependency ordering of the action
  ActionOrder order() const final {{ return ActionOrder::{actionorder}; }}
}};

#if !CELER_USE_DEVICE
inline void {clsname}::execute(CoreDeviceRef const&) const
{{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}}
#endif

//---------------------------------------------------------------------------//
}}  // namespace generated
}}  // namespace celeritas
"""

CC_TEMPLATE = CLIKE_TOP + """\
#include "{clsname}.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/global/KernelContextException.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "../detail/{clsname}Impl.hh" // IWYU pragma: associated

namespace celeritas
{{
namespace generated
{{
void {clsname}::execute(CoreHostRef const& data) const
{{
    CELER_EXPECT(data);

    MultiExceptionHandler capture_exception;
    auto launch = make_track_launcher(data, detail::{func}_track);
    #pragma omp parallel for
    for (size_type i = 0; i < data.states.size(); ++i)
    {{
        CELER_TRY_HANDLE_CONTEXT(
            launch(TrackSlotId{{i}}),
            capture_exception,
            KernelContextException(data, TrackSlotId{{i}}, this->label()));
    }}
    log_and_rethrow(std::move(capture_exception));
}}

}}  // namespace generated
}}  // namespace celeritas
"""

CU_TEMPLATE = CLIKE_TOP + """\
#include "{clsname}.hh"

#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/global/TrackLauncher.hh"
#include "../detail/{clsname}Impl.hh"

namespace celeritas
{{
namespace generated
{{
namespace
{{
__global__ void{launch_bounds}{func}_kernel(CoreDeviceRef const data
)
{{
    auto tid = KernelParamCalculator::thread_id();
    if (!(tid < data.states.size()))
        return;

    auto launch = make_track_launcher(data, detail::{func}_track);
    launch(tid);
}}
}}  // namespace

void {clsname}::execute(CoreDeviceRef const& data) const
{{
    CELER_EXPECT(data);
    CELER_LAUNCH_KERNEL({func},
                        celeritas::device().default_block_size(),
                        data.states.size(),
                        data);
}}

}}  // namespace generated
}}  // namespace celeritas
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
    subs['launch_bounds'] = make_launch_bounds(subs['func'])
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
        '--class', dest='clsname',
        help='CamelCase name of the class prefix')
    parser.add_argument(
        '--func',
        help='snake_case name of the function')
    parser.add_argument(
        '--actionorder',
        help='Inter-kernel dependency order')

    kwargs = vars(parser.parse_args())
    for ext in ['hh', 'cc', 'cu']:
        generate(ext=ext, **kwargs)

if __name__ == '__main__':
    main()
