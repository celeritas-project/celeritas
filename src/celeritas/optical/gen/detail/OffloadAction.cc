//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadAction.cc
//---------------------------------------------------------------------------//
#include "OffloadAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "CherenkovOffloadExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "ScintOffloadExecutor.hh"
#include "../CherenkovParams.hh"
#include "../ScintillationParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
template<GeneratorType G>
std::shared_ptr<OffloadAction<G>>
OffloadAction<G>::make_and_insert(CoreParams const& core, Input&& input)
{
    CELER_EXPECT(input);
    ActionRegistry& actions = *core.action_reg();
    auto result = std::make_shared<OffloadAction<G>>(actions.next_id(),
                                                     std::move(input));

    actions.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, aux IDs, and optical properties.
 */
template<GeneratorType G>
OffloadAction<G>::OffloadAction(ActionId id, Input&& inp)
    : action_id_(id), data_(std::move(inp))
{
    CELER_EXPECT(action_id_);
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
template<GeneratorType G>
void OffloadAction<G>::step(CoreParams const& params, CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
template<GeneratorType G>
void OffloadAction<G>::step(CoreParams const& params,
                            CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical distribution data post-step.
 */
template<GeneratorType G>
template<MemSpace M>
void OffloadAction<G>::step_impl(CoreParams const& core_params,
                                 CoreState<M>& core_state) const
{
    auto& gen_state = get<GeneratorState<M>>(core_state.aux(), data_.gen_id);
    auto& buffer = gen_state.store.ref().distributions;
    auto& buffer_size = gen_state.counters.buffer_size;

    CELER_VALIDATE(buffer_size + core_state.size() <= buffer.size(),
                   << "insufficient capacity (" << buffer.size()
                   << ") for buffered optical photon distribution data (total "
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
        = get<optical::CoreState<M>>(core_state.aux(), data_.optical_id);
    optical_state.counters().num_pending += count_num_photons(
        buffer, start, buffer_size, core_state.stream_id());
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical distribution data post-step.
 */
template<GeneratorType G>
void OffloadAction<G>::offload(CoreParams const& core_params,
                               CoreStateHost& core_state) const
{
    auto& step = core_state.aux_data<OffloadStepStateData>(data_.step_id);
    auto& gen_state = get<GeneratorState<MemSpace::native>>(core_state.aux(),
                                                            data_.gen_id);
    TrackExecutor execute{core_params.ptr<MemSpace::native>(),
                          core_state.ptr(),
                          Executor{data_.material->host_ref(),
                                   data_.shared->host_ref(),
                                   gen_state.store.ref(),
                                   step,
                                   gen_state.counters.buffer_size}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<GeneratorType G>
void OffloadAction<G>::offload(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class OffloadAction<GeneratorType::cherenkov>;
template class OffloadAction<GeneratorType::scintillation>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
