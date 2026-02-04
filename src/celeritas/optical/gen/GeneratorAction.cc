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
template<MemSpace M>
auto make_state(StreamId stream, size_type size)
{
    using StoreT = CollectionStateStore<GeneratorStateData, M>;

    auto result = std::make_unique<GeneratorState<M>>();
    result->store = StoreT{stream, size};

    CELER_ENSURE(*result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<GeneratorAction>
GeneratorAction::make_and_insert(CoreParams const& params, size_type capacity)
{
    ActionRegistry& actions = *params.action_reg();
    AuxParamsRegistry& aux = *params.aux_reg();
    GeneratorRegistry& gen = *params.gen_reg();
    auto result = std::make_shared<GeneratorAction>(
        actions.next_id(), aux.next_id(), gen.next_id(), capacity);

    actions.insert(result);
    aux.insert(result);
    gen.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action ID, data IDs, and optical properties.
 */
GeneratorAction::GeneratorAction(ActionId id,
                                 AuxId aux_id,
                                 GeneratorId gen_id,
                                 size_type capacity)
    : GeneratorBase(id,
                    aux_id,
                    gen_id,
                    "optical-generate",
                    "generate Cherenkov or scintillation photons from optical "
                    "distribution data")
    , initial_capacity_(capacity)
{
    CELER_EXPECT(initial_capacity_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto GeneratorAction::create_state(MemSpace m, StreamId id, size_type) const
    -> UPState
{
    if (m == MemSpace::host)
    {
        return make_state<MemSpace::host>(id, initial_capacity_);
    }
    else if (m == MemSpace::device)
    {
        return make_state<MemSpace::device>(id, initial_capacity_);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Add user-provided host distribution data.
 */
void GeneratorAction::insert(CoreStateBase& state, SpanConstData data) const
{
    if (auto* s = dynamic_cast<CoreState<MemSpace::host>*>(&state))
    {
        return this->insert_impl(*s, data);
    }
    else if (auto* s = dynamic_cast<CoreState<MemSpace::device>*>(&state))
    {
        return this->insert_impl(*s, data);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void GeneratorAction::step(CoreParams const& params, CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void GeneratorAction::step(CoreParams const& params,
                           CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Add distributions to the aux state.
 */
template<MemSpace M>
void GeneratorAction::insert_impl(CoreState<M>& state, SpanConstData data) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<GeneratorState<M>>(*state.aux(), this->aux_id());

    if (aux_state.counters.buffer_size != 0)
    {
        CELER_NOT_IMPLEMENTED("multiple consecutive distribution insertions");
    }

    if (aux_state.store.size() < data.size())
    {
        // Reallocate with enough capacity
        aux_state.store = CollectionStateStore<GeneratorStateData, M>{
            state.stream_id(), static_cast<size_type>(data.size())};
    }

    // Update counters and copy distributions to aux state
    aux_state.counters.buffer_size = data.size();
    Copier<GeneratorDistributionData, M> copy_to_aux{aux_state.distributions(),
                                                     state.stream_id()};
    copy_to_aux(MemSpace::host, data);
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons from distribution data.
 */
template<MemSpace M>
void GeneratorAction::step_impl(CoreParams const& params,
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

    if (state.sync_get_counters().num_vacancies > 0 && counters.num_pending > 0)
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
void GeneratorAction::generate(CoreParams const& params,
                               CoreStateHost& state) const
{
    CELER_EXPECT(params.cherenkov() || params.scintillation());
    CELER_EXPECT(state.aux());

    auto& aux_state
        = get<GeneratorState<MemSpace::native>>(*state.aux(), this->aux_id());
    size_type num_gen = min(state.sync_get_counters().num_vacancies,
                            aux_state.counters.num_pending);
    {
        // Generate optical photons in vacant track slots
        detail::GeneratorExecutor execute{params.ptr<MemSpace::native>(),
                                          state.ptr(),
                                          params.host_ref().cherenkov,
                                          params.host_ref().scintillation,
                                          aux_state.store.ref(),
                                          aux_state.counters.buffer_size};
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
void GeneratorAction::generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
