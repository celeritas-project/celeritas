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
#include "corecel/sys/KernelLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/action/ActionLauncher.hh"

#include "GeneratorExecutor.hh"
#include "OpticalGenAlgorithms.hh"
#include "UpdateSumExecutor.hh"
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
GeneratorAction<G>::make_and_insert(::celeritas::CoreParams const& core_params,
                                    optical::CoreParams const& params,
                                    Input&& input)
{
    CELER_EXPECT(input);
    ActionRegistry& actions = *params.action_reg();
    AuxParamsRegistry& aux = *core_params.aux_reg();
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
void GeneratorAction<G>::step(optical::CoreParams const& params,
                              CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
template<GeneratorType G>
void GeneratorAction<G>::step(optical::CoreParams const& params,
                              CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons from distribution data.
 */
template<GeneratorType G>
template<MemSpace M>
void GeneratorAction<G>::step_impl(optical::CoreParams const& params,
                                   optical::CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<GeneratorState<M>>(*state.aux(), aux_id_);

    if (aux_state.buffer_size == 0)
    {
        // No new photons from this generating process
        return;
    }

    if (aux_state.num_generated == 0)
    {
        // On the first step iteration, calculate the cumulative sum of the
        // number of photons in the buffered distributions. These values are
        // used to determine which thread will generate photons from which
        // distribution
        aux_state.num_pending
            = inclusive_scan_photons(aux_state.store.ref().distributions,
                                     aux_state.store.ref().offsets,
                                     aux_state.buffer_size,
                                     state.stream_id());
    }

    auto& counters = state.counters();
    size_type num_gen = min(counters.num_vacancies, aux_state.num_pending);
    if (num_gen > 0)
    {
        // Generate the optical photons from the distribution data
        this->generate(params, state);

        // Update the optical core state counters
        counters.num_pending -= num_gen;
        counters.num_generated += num_gen;
        counters.num_vacancies -= num_gen;

        // Update the generator counters and statistics
        aux_state.num_pending -= num_gen;
        aux_state.num_generated += num_gen;
        aux_state.accum.photons += num_gen;
        if (aux_state.num_pending == 0)
        {
            // Reset the buffer size and number of photons generated
            aux_state.accum.distributions += aux_state.buffer_size;
            aux_state.buffer_size = 0;
            aux_state.num_generated = 0;
        }
    }
    counters.num_active = state.size() - counters.num_vacancies;
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photons.
 */
template<GeneratorType G>
void GeneratorAction<G>::generate(optical::CoreParams const& params,
                                  CoreStateHost& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state
        = get<GeneratorState<MemSpace::native>>(*state.aux(), aux_id_);
    size_type num_gen
        = min(state.counters().num_vacancies, aux_state.num_pending);
    {
        // Generate optical photons in vacant track slots
        detail::GeneratorExecutor<G> execute{params.ptr<MemSpace::native>(),
                                             state.ptr(),
                                             data_.material->host_ref(),
                                             data_.shared->host_ref(),
                                             aux_state.store.ref(),
                                             aux_state.buffer_size,
                                             state.counters()};
        celeritas::optical::launch_action(num_gen, execute);
    }
    {
        // Update the cumulative sum of the number of photons per distribution
        // according to how many were generated
        detail::UpdateSumExecutor execute{aux_state.store.ref(), num_gen};
        launch_kernel(aux_state.buffer_size, execute);
    }
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
template<GeneratorType G>
void GeneratorAction<G>::generate(optical::CoreParams const&,
                                  CoreStateDevice&) const
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
