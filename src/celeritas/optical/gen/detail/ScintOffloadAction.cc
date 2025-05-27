//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/ScintOffloadAction.cc
//---------------------------------------------------------------------------//
#include "ScintOffloadAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreState.hh"

#include "OpticalGenAlgorithms.hh"
#include "ScintOffloadExecutor.hh"
#include "../ScintillationParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data ID, optical properties.
 */
ScintOffloadAction::ScintOffloadAction(ActionId action_id,
                                       AuxId step_id,
                                       AuxId gen_id,
                                       AuxId optical_id,
                                       SPConstScintillation scintillation)
    : action_id_(action_id)
    , step_id_{step_id}
    , gen_id_{gen_id}
    , optical_id_{optical_id}
    , scintillation_(std::move(scintillation))
{
    CELER_EXPECT(action_id_);
    CELER_EXPECT(step_id_);
    CELER_EXPECT(gen_id_);
    CELER_EXPECT(optical_id_);
    CELER_EXPECT(scintillation_);
}

//---------------------------------------------------------------------------//
/*!
 * Descriptive name of the action.
 */
std::string_view ScintOffloadAction::description() const
{
    return "generate scintillation optical distribution data";
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void ScintOffloadAction::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void ScintOffloadAction::step(CoreParams const& params,
                              CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<MemSpace M>
void ScintOffloadAction::step_impl(CoreParams const& core_params,
                                   CoreState<M>& core_state) const
{
    auto& gen_state = get<GeneratorState<M>>(core_state.aux(), gen_id_);
    auto& buffer = gen_state.store.ref().distributions;
    auto& buffer_size = gen_state.buffer_size;

    CELER_VALIDATE(buffer_size + core_state.size() <= buffer.size(),
                   << "insufficient capacity (" << buffer.size()
                   << ") for buffered scintillation distribution data (total "
                      "capacity requirement of "
                   << buffer_size + core_state.size() << ")");

    // Generate the optical distribution data
    this->offload(core_params, core_state);

    // Compact the buffer, returning the total number of valid distributions
    size_type start = buffer_size;
    buffer_size = remove_if_invalid(
        buffer, start, start + core_state.size(), core_state.stream_id());

    // Count the number of optical photons that would be generated from the
    // distributions created in this step
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), optical_id_);
    optical_state.counters().num_pending += count_num_photons(
        buffer, start, buffer_size, core_state.stream_id());
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical distribution data post-step.
 */
void ScintOffloadAction::offload(CoreParams const& core_params,
                                 CoreStateHost& core_state) const
{
    auto& step_state
        = get<OffloadStepState<MemSpace::native>>(core_state.aux(), step_id_);
    auto& gen_state
        = get<GeneratorState<MemSpace::native>>(core_state.aux(), gen_id_);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::ScintOffloadExecutor{scintillation_->host_ref(),
                                     gen_state.store.ref(),
                                     step_state.store.ref(),
                                     gen_state.buffer_size}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ScintOffloadAction::offload(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
