//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.cu
//---------------------------------------------------------------------------//
#include "ExtendFromSecondariesAction.hh"

#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/LocateAliveExecutor.hh"
#include "detail/ProcessSecondariesExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to locate alive particles.
 *
 * This fills the TrackInit \c vacancies and \c secondary_counts arrays.
 */
void ExtendFromSecondariesAction::locate_alive(CoreParams const& core_params,
                                               CoreStateDevice& core_state) const
{
    using Executor = detail::LocateAliveExecutor;
    static ActionLauncher<Executor> launch(*this, "locate-alive");
    launch(core_state,
           Executor{core_params.ptr<MemSpace::native>(), core_state.ptr()});
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to create track initializers from secondary particles.
 */
void ExtendFromSecondariesAction::process_secondaries(
    CoreParams const& core_params, CoreStateDevice& core_state) const
{
    using Executor = detail::ProcessSecondariesExecutor;
    static ActionLauncher<Executor> launch(*this, "process-secondaries");
    launch(core_state,
           Executor{core_params.ptr<MemSpace::native>(),
                    core_state.ptr(),
                    core_state.counters()});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
