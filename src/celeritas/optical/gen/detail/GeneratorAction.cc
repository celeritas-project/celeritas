//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorAction.cc
//---------------------------------------------------------------------------//
#include "GeneratorAction.hh"

#include <algorithm>

#include "corecel/Assert.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "GeneratorExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "../CherenkovGenerator.hh"
#include "../CherenkovParams.hh"
#include "../ScintillationGenerator.hh"
#include "../ScintillationParams.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
//! Construct a state
template<class P, MemSpace M>
auto make_state(P const& params, StreamId stream, size_type size)
{
    using StoreT = CollectionStateStore<GeneratorStateData, M>;

    auto result = std::make_unique<GeneratorState<M>>();
    result->store = StoreT{params.host_ref(), stream, size};

    CELER_ENSURE(*result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
template<GeneratorType G>
std::shared_ptr<GeneratorAction<G>>
GeneratorAction<G>::make_and_insert(CoreParams const& core, Input&& input)
{
    CELER_EXPECT(input);
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result = std::make_shared<GeneratorAction<G>>(
        actions.next_id(), aux.next_id(), std::move(input));

    actions.insert(result);
    aux.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data IDs, and optical properties.
 */
template<GeneratorType G>
GeneratorAction<G>::GeneratorAction(ActionId id, AuxId aux_id, Input&& inp)
    : action_id_(id), aux_id_(aux_id), data_(std::move(inp))
{
    CELER_EXPECT(action_id_);
    CELER_EXPECT(aux_id_);
    CELER_EXPECT(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
template<GeneratorType G>
auto GeneratorAction<G>::create_state(MemSpace m, StreamId id, size_type) const
    -> UPState
{
    using Params = typename TraitsT::Params;
    if (m == MemSpace::host)
    {
        return make_state<Params, MemSpace::host>(
            *data_.shared, id, data_.capacity);
    }
    else if (m == MemSpace::device)
    {
        return make_state<Params, MemSpace::device>(
            *data_.shared, id, data_.capacity);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
template<GeneratorType G>
void GeneratorAction<G>::step(CoreParams const& params,
                              CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
template<GeneratorType G>
void GeneratorAction<G>::step(CoreParams const& params,
                              CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical track initializers from distribution data.
 */
template<GeneratorType G>
template<MemSpace M>
void GeneratorAction<G>::step_impl(CoreParams const& core_params,
                                   CoreState<M>& core_state) const
{
    auto& aux_state = get<GeneratorState<M>>(core_state.aux(), aux_id_);
    auto& optical_state
        = get<optical::CoreState<M>>(core_state.aux(), data_.optical_id);

    auto& photons = optical_state.counters().num_initializers;
    auto& num_new_photons = optical_state.counters().num_pending;

    if (photons + num_new_photons < data_.auto_flush)
    {
        // Below the threshold for launching the optical loop
        return;
    }

    auto initializers_size = optical_state.ref().init.initializers.size();
    CELER_VALIDATE(photons + num_new_photons <= initializers_size,
                   << "insufficient capacity (" << initializers_size
                   << ") for optical photon initializers (total capacity "
                      "requirement of "
                   << photons + num_new_photons << " and current size "
                   << photons << ")");

    if (aux_state.buffer_size == 0)
    {
        // No new photons
        return;
    }

    // Calculate the cumulative sum of the number of photons in the buffered
    // distributions. These values are used to determine which thread will
    // generate initializers from which distribution
    auto count = inclusive_scan_photons(aux_state.store.ref().distributions,
                                        aux_state.store.ref().offsets,
                                        aux_state.buffer_size,
                                        core_state.stream_id());
    optical_state.counters().num_generated += count;

    // Generate the optical photon initializers from the distribution data
    this->generate(core_params, core_state);

    // Update cumulative statistics
    aux_state.accum.distributions += aux_state.buffer_size;
    aux_state.accum.photons += count;

    photons += count;
    num_new_photons -= count;
    aux_state.buffer_size = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photon initializers.
 */
template<GeneratorType G>
void GeneratorAction<G>::generate(CoreParams const& core_params,
                                  CoreStateHost& core_state) const
{
    auto& aux_state
        = get<GeneratorState<MemSpace::native>>(core_state.aux(), aux_id_);
    auto& optical_state = get<optical::CoreState<MemSpace::native>>(
        core_state.aux(), data_.optical_id);

    TrackExecutor execute{
        core_params.ptr<MemSpace::native>(),
        core_state.ptr(),
        detail::GeneratorExecutor<G>{core_state.ptr(),
                                     data_.material->host_ref(),
                                     data_.shared->host_ref(),
                                     aux_state.store.ref(),
                                     optical_state.ptr(),
                                     aux_state.buffer_size,
                                     optical_state.counters()}};
    launch_action(*this, core_params, core_state, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<GeneratorType G>
void GeneratorAction<G>::generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class GeneratorAction<GeneratorType::cherenkov>;
template class GeneratorAction<GeneratorType::scintillation>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
