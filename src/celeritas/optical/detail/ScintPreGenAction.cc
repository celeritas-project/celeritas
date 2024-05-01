//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/ScintPreGenAction.cc
//---------------------------------------------------------------------------//
#include "ScintPreGenAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/ScintillationParams.hh"

#include "OpticalGenAlgorithms.hh"
#include "OpticalGenStorage.hh"
#include "ScintPreGenExecutor.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, optical properties, and storage.
 */
ScintPreGenAction::ScintPreGenAction(ActionId id,
                                     SPConstScintillation scintillation,
                                     SPGenStorage storage)
    : id_(id)
    , scintillation_(std::move(scintillation))
    , storage_(std::move(storage))
{
    CELER_EXPECT(id_);
    CELER_EXPECT(scintillation_);
    CELER_EXPECT(storage_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view ScintPreGenAction::description() const
{
    return "generate scintillation optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void ScintPreGenAction::execute(CoreParams const& params,
                                CoreStateHost& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void ScintPreGenAction::execute(CoreParams const& params,
                                CoreStateDevice& state) const
{
    this->execute_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<MemSpace M>
void ScintPreGenAction::execute_impl(CoreParams const& core_params,
                                     CoreState<M>& core_state) const
{
    size_type state_size = core_state.size();
    StreamId stream = core_state.stream_id();
    auto& buffer_size = storage_->size[stream.get()];
    auto const& state = storage_->obj.state<M>(stream, state_size);

    CELER_VALIDATE(
        buffer_size.scintillation + state_size <= state.scintillation.size(),
        << "insufficient capacity (" << state.scintillation.size()
        << ") for buffered scintillation distribution data (total "
           "capacity requirement of "
        << buffer_size.scintillation + state_size << ")");

    // Generate the optical distribution data
    this->pre_generate(core_params, core_state);

    // Compact the buffer
    buffer_size.scintillation = remove_if_invalid(
        state.scintillation, buffer_size.scintillation, state_size, stream);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical distribution data post-step.
 */
void ScintPreGenAction::pre_generate(CoreParams const& core_params,
                                     CoreStateHost& core_state) const
{
    TrackExecutor execute{core_params.ptr<MemSpace::native>(),
                          core_state.ptr(),
                          detail::ScintPreGenExecutor{
                              scintillation_->host_ref(),
                              storage_->obj.state<MemSpace::native>(
                                  core_state.stream_id(), core_state.size()),
                              storage_->size[core_state.stream_id().get()]}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ScintPreGenAction::pre_generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
