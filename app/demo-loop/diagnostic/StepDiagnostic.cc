//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StepDiagnostic.cc
//---------------------------------------------------------------------------//
#include "StepDiagnostic.hh"

using namespace celeritas;

namespace demo_loop
{
//---------------------------------------------------------------------------//
/*!
 * Count the steps per track for each particle type.
 */
void count_steps(const CoreParamsHostRef&              params,
                 const CoreStateHostRef&               states,
                 StepDiagnosticDataRef<MemSpace::host> data)
{
    StepLauncher<MemSpace::host> launch(params, states, data);
    for (auto tid : range(ThreadId{states.size()}))
    {
        launch(tid);
    }
}
} // namespace demo_loop
