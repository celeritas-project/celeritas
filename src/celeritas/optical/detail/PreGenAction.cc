//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/PreGenAction.cc
//---------------------------------------------------------------------------//
#include "PreGenAction.hh"

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CerenkovParams.hh"
#include "celeritas/optical/OpticalPropertyParams.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "GenStorage.hh"
#include "PreGenExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Capture construction arguments.
 */
PreGenAction::PreGenAction(ActionId id, SPGenStorage storage)
    : id_(id), storage_(std::move(storage))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string PreGenAction::description() const
{
    return "generate optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
void PreGenAction::execute(CoreParams const& params, CoreStateHost& state) const
{
    auto const& gen_state = storage_->obj.state<MemSpace::native>(
        state.stream_id(), state.size());
    auto execute = TrackExecutor{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenExecutor{storage_->obj.params<MemSpace::native>(),
                               gen_state}};
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void PreGenAction::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
