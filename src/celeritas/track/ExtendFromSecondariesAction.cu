//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromSecondariesAction.cu
//---------------------------------------------------------------------------//
#include "ExtendFromSecondariesAction.hh"

#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

#include "detail/LocateAliveExecutor.hh"
#include "detail/ProcessSecondariesExecutor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Warm up asynchronous allocation.
 *
 * This just calls MallocAsync before the first step, since it's used by
 * \c detail::remove_if_alive under the hood.
 */
void ExtendFromSecondariesAction::begin_run(CoreParams const&,
                                            CoreStateDevice& core_state)
{
    ScopedProfiling profile_this{this->label()};
    Stream& s = device().stream(core_state.stream_id());
    void* p = s.malloc_async(core_state.size() * sizeof(size_type));
    s.free_async(p);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a kernel to locate alive particles.
 *
 * This fills the TrackInit \c vacancies and \c secondary_counts arrays.
 */
void ExtendFromSecondariesAction::locate_alive(CoreParams const& core_params,
                                               CoreStateDevice& core_state) const
{
    ScopedProfiling profile_this{"locate-alive"};
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
    ScopedProfiling profile_this{"process-secondaries"};
    using Executor = detail::ProcessSecondariesExecutor;
    static ActionLauncher<Executor> launch(*this, "process-secondaries");
    launch(core_state,
           Executor{core_params.ptr<MemSpace::native>(),
                    core_state.ptr(),
                    core_state.counters()});
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
