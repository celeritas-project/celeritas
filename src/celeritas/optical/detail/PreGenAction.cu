//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenAction.cu
//---------------------------------------------------------------------------//
#include "PreGenAction.hh"

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.device.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "GenStorage.hh"
#include "PreGenExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
void PreGenAction::execute(CoreParams const& params,
                           CoreStateDevice& state) const
{
    auto& gen_state = storage_->obj.state<MemSpace::native>(state.stream_id(),
                                                            state.size());
    auto execute = TrackExecutor{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenExecutor{storage_->obj.params<MemSpace::native>(),
                               gen_state}};
    static ActionLauncher<decltype(execute)> const launch_kernel(*this);
    launch_kernel(state, execute);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
