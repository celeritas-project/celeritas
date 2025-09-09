//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorAction.cc
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
#include "celeritas/phys/GeneratorRegistry.hh"

#include "CherenkovGenerator.hh"
#include "CherenkovParams.hh"
#include "ScintillationGenerator.hh"
#include "ScintillationParams.hh"

#include "detail/GeneratorAlgorithms.hh"
#include "detail/GeneratorExecutor.hh"
#include "detail/UpdateSumExecutor.hh"

namespace celeritas
{
namespace optical
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
                                    CoreParams const& params,
                                    Input&& input)
{
    CELER_EXPECT(input);
    ActionRegistry& actions = *params.action_reg();
    AuxParamsRegistry& aux = *core_params.aux_reg();
    GeneratorRegistry& gen = *params.gen_reg();
    auto result = std::make_shared<GeneratorAction<G>>(
        actions.next_id(), aux.next_id(), gen.next_id(), std::move(input));

    actions.insert(result);
    aux.insert(result);
    gen.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data IDs, and optical properties.
 */
template<GeneratorType G>
GeneratorAction<G>::GeneratorAction(ActionId id,
                                    AuxId aux_id,
                                    GeneratorId gen_id,
                                    Input&& inp)
    : GeneratorBase(id, aux_id, gen_id, TraitsT::label, TraitsT::description)
    , data_(std::move(inp))
{
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
 * Generate optical photons from distribution data.
 */
template<GeneratorType G>
template<MemSpace M>
void GeneratorAction<G>::step_impl(CoreParams const& params,
                                   CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<GeneratorState<M>>(*state.aux(), this->aux_id());
    auto& counters = aux_state.counters;

    if (counters.num_generated == 0 && counters.buffer_size > 0)
    {
        // If this process created photons, on the first step iteration
        // calculate the cumulative sum of the number of photons in the
        // buffered distributions. These values are used to determine which
        // thread will generate photons from which distribution
        counters.num_pending = detail::inclusive_scan_photons(
            aux_state.store.ref().distributions,
            aux_state.store.ref().offsets,
            counters.buffer_size,
            state.stream_id());
    }

    if (state.counters().num_vacancies > 0 && counters.num_pending > 0)
    {
        // Generate the optical photons from the distribution data
        this->generate(params, state);
    }

    // Update the generator and optical core state counters
    this->update_counters(state);

    // If there are no more tracks to generate, reset the buffer size and
    // number of photons generated
    if (counters.num_pending == 0)
    {
        aux_state.accum.buffer_size += counters.buffer_size;
        counters = {};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to generate optical photons.
 */
template<GeneratorType G>
void GeneratorAction<G>::generate(CoreParams const& params,
                                  CoreStateHost& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state
        = get<GeneratorState<MemSpace::native>>(*state.aux(), this->aux_id());
    size_type num_gen
        = min(state.counters().num_vacancies, aux_state.counters.num_pending);
    {
        // Generate optical photons in vacant track slots
        detail::GeneratorExecutor<G> execute{params.ptr<MemSpace::native>(),
                                             state.ptr(),
                                             data_.material->host_ref(),
                                             data_.shared->host_ref(),
                                             aux_state.store.ref(),
                                             aux_state.counters.buffer_size,
                                             state.counters()};
        launch_action(num_gen, execute);
    }
    {
        // Update the cumulative sum of the number of photons per distribution
        // according to how many were generated
        detail::UpdateSumExecutor execute{aux_state.store.ref(), num_gen};
        launch_kernel(aux_state.counters.buffer_size, execute);
    }
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
}  // namespace optical
}  // namespace celeritas
