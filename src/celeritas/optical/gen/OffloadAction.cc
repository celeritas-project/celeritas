//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/OffloadAction.cc
//---------------------------------------------------------------------------//
#include "OffloadAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/MaterialParams.hh"  // IWYU pragma: keep

#include "CherenkovParams.hh"  // IWYU pragma: keep
#include "ScintillationParams.hh"  // IWYU pragma: keep

#include "detail/CherenkovOffloadExecutor.hh"  // IWYU pragma: keep
#include "detail/OffloadAlgorithms.hh"
#include "detail/ScintOffloadExecutor.hh"  // IWYU pragma: keep

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
template<GeneratorType G>
std::shared_ptr<OffloadAction<G>> OffloadAction<G>::make_and_insert(
    CoreParams const& core, Input&& input)
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
    CELER_EXPECT(data_.pre_post_step_id || G == GeneratorType::cherenkov);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
template<GeneratorType G>
void OffloadAction<G>::step(CoreParams const& params,
                            CoreStateHost& state) const
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
    auto& gen_state
        = get<optical::GeneratorState<M>>(core_state.aux(), data_.gen_id);

    // Grow the buffer if needed to hold the distributions from this step. The
    // buffer size is only known on device, so use an upper bound.
    size_type bound = gen_state.buffer_bound();
    detail::reserve_distributions(&gen_state.store,
                                  bound,
                                  bound + core_state.size(),
                                  core_state.stream_id(),
                                  this->label());
    auto& buffer = gen_state.store.ref().distributions;
    CELER_ASSERT(bound + core_state.size() <= buffer.size());

    // Generate the optical distribution data
    this->offload(core_params, core_state);

    // Append the valid distributions to the buffer and add the number of
    // optical photons that would be generated from them to the number of
    // pending tracks
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), data_.optical_id);
    detail::append_distributions(optical_state,
                                 buffer,
                                 gen_state.device_counters(),
                                 core_state.size(),
                                 core_state.stream_id());

    // Up to one distribution per track slot was added
    gen_state.grow_bound(core_state.size());
    gen_state.needs_scan = true;
    gen_state.copy_async();
}

//---------------------------------------------------------------------------//
/*!
 * Get the pre-step auxiliary data for this action.
 */
template<GeneratorType G>
template<MemSpace M>
typename OffloadAction<G>::PreTraitsT::template Data<Ownership::reference, M>&
OffloadAction<G>::get_pre_step_data(CoreState<M>& state) const
{
    return state.template aux_data<PreTraitsT::template Data>(
        data_.pre_step_id);
}

//---------------------------------------------------------------------------//
/*!
 * Get the pre-post-step auxiliary data for this action.
 */
template<GeneratorType G>
template<MemSpace M>
typename OffloadAction<G>::PostTraitsT::template Data<Ownership::reference, M>&
OffloadAction<G>::get_post_step_data(CoreState<M>& state) const
{
    return state.template aux_data<PostTraitsT::template Data>(
        data_.pre_post_step_id);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical distribution data post-step.
 */
template<GeneratorType G>
void OffloadAction<G>::offload(CoreParams const& core_params,
                               CoreStateHost& core_state) const
{
    auto& pre_step = this->get_pre_step_data(core_state);
    auto& gen_state = get<optical::GeneratorState<MemSpace::native>>(
        core_state.aux(), data_.gen_id);
    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        Executor{data_.material->host_ref(),
                 data_.shared->host_ref(),
                 gen_state.store.ref(),
                 pre_step,
                 (G == GeneratorType::scintillation)
                     ? this->get_post_step_data(core_state)
                     : NativeRef<PostTraitsT::template Data>{},
                 gen_state.device_counters()}};
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

template typename OffloadAction<
    GeneratorType::cherenkov>::PreTraitsT::template Data<Ownership::reference,
                                                         MemSpace::device>&
OffloadAction<GeneratorType::cherenkov>::get_pre_step_data(
    CoreState<MemSpace::device>&) const;
template typename OffloadAction<
    GeneratorType::scintillation>::PreTraitsT::template Data<Ownership::reference,
                                                             MemSpace::device>&
OffloadAction<GeneratorType::scintillation>::get_pre_step_data(
    CoreState<MemSpace::device>&) const;

template typename OffloadAction<
    GeneratorType::cherenkov>::PostTraitsT::template Data<Ownership::reference,
                                                          MemSpace::device>&
OffloadAction<GeneratorType::cherenkov>::get_post_step_data(
    CoreState<MemSpace::device>&) const;
template typename OffloadAction<
    GeneratorType::scintillation>::PostTraitsT::template Data<Ownership::reference,
                                                              MemSpace::device>&
OffloadAction<GeneratorType::scintillation>::get_post_step_data(
    CoreState<MemSpace::device>&) const;

//---------------------------------------------------------------------------//
}  // namespace celeritas
