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

#include "GenStorage.hh"
#include "PreGenExecutor.hh"
#include "PreGenGatherExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Capture construction arguments.
 */
template<StepPoint P>
PreGenAction<P>::PreGenAction(ActionId id, SPGenStorage storage)
    : id_(id), storage_(std::move(storage))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
template<StepPoint P>
std::string PreGenAction<P>::description() const
{
    std::string result = "gather ";
    result += P == StepPoint::pre ? "pre" : "post";
    result += "-step data to generate optical distributions";
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
template<>
void PreGenAction<StepPoint::pre>::execute(CoreParams const& params,
                                           CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenGatherExecutor{storage_->obj.state<MemSpace::native>(
            state.stream_id(), state.size())});
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data.
 */
template<>
void PreGenAction<StepPoint::post>::execute(CoreParams const& params,
                                            CoreStateHost& state) const
{
    auto execute = make_active_track_executor(
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::PreGenExecutor{storage_->obj.params<MemSpace::native>(),
                               storage_->obj.state<MemSpace::native>(
                                   state.stream_id(), state.size())});
    launch_action(*this, params, state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<StepPoint P>
void PreGenAction<P>::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class PreGenAction<StepPoint::pre>;
template class PreGenAction<StepPoint::post>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
