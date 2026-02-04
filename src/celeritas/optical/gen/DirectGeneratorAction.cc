//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/DirectGeneratorAction.cc
//---------------------------------------------------------------------------//
#include "DirectGeneratorAction.hh"

#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/KernelLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/CoreTrackData.hh"
#include "celeritas/optical/action/ActionLauncher.hh"
#include "celeritas/phys/GeneratorRegistry.hh"

#include "detail/DirectGeneratorExecutor.hh"
#include "detail/GeneratorAlgorithms.hh"

namespace celeritas
{
namespace optical
{
namespace
{
//---------------------------------------------------------------------------//
//! Construct a state.
template<MemSpace M>
auto make_state(StreamId stream, size_type size)
{
    using StoreT = CollectionStateStore<DirectGeneratorStateData, M>;

    auto result = std::make_unique<DirectGeneratorState<M>>();
    result->store = StoreT{stream, size};

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<DirectGeneratorAction>
DirectGeneratorAction::make_and_insert(CoreParams const& params)
{
    ActionRegistry& actions = *params.action_reg();
    AuxParamsRegistry& aux = *params.aux_reg();
    GeneratorRegistry& gen = *params.gen_reg();
    auto result = std::make_shared<DirectGeneratorAction>(
        actions.next_id(), aux.next_id(), gen.next_id());
    actions.insert(result);
    aux.insert(result);
    gen.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with action and data IDs.
 */
DirectGeneratorAction::DirectGeneratorAction(ActionId id,
                                             AuxId aux_id,
                                             GeneratorId gen_id)
    : GeneratorBase(id,
                    aux_id,
                    gen_id,
                    "generate-direct",
                    "directly generate optical photon primaries")
{
}

//---------------------------------------------------------------------------//
/*!
 * Insert user-provided host initializer data.
 */
void DirectGeneratorAction::insert(CoreStateBase& state,
                                   SpanConstData data) const
{
    if (auto* s = dynamic_cast<CoreStateHost*>(&state))
    {
        return this->insert_impl(*s, data);
    }
    else if (auto* s = dynamic_cast<CoreStateDevice*>(&state))
    {
        return this->insert_impl(*s, data);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void DirectGeneratorAction::step(CoreParams const& params,
                                 CoreStateHost& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void DirectGeneratorAction::step(CoreParams const& params,
                                 CoreStateDevice& state) const
{
    this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Add initializers to the aux state.
 */
template<MemSpace M>
void DirectGeneratorAction::insert_impl(CoreState<M>& state,
                                        SpanConstData data) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state
        = get<DirectGeneratorState<M>>(*state.aux(), this->aux_id());

    if (aux_state.counters.buffer_size != 0)
    {
        CELER_NOT_IMPLEMENTED(
            "multiple consecutive direct photon insertions not supported");
    }

    if (aux_state.store.size() < data.size())
    {
        // Reallocate with enough capacity
        aux_state.store = CollectionStateStore<DirectGeneratorStateData, M>{
            state.stream_id(), static_cast<size_type>(data.size())};
    }

    // Update counters and copy distributions to aux state storage
    aux_state.counters.buffer_size = data.size();
    aux_state.counters.num_pending = data.size();
    auto counters = state.sync_get_counters();
    counters.num_pending += data.size();
    state.sync_put_counters(counters);
    Copier<TrackInitializer, M> copy_to_aux{aux_state.initializers(),
                                            state.stream_id()};

    copy_to_aux(MemSpace::host, data);
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto DirectGeneratorAction::create_state(MemSpace m,
                                         StreamId stream,
                                         size_type size) const -> UPState
{
    if (m == MemSpace::host)
    {
        return make_state<MemSpace::host>(stream, size);
    }
    else if (m == MemSpace::device)
    {
        return make_state<MemSpace::device>(stream, size);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Generate optical photons directly from initializers.
 */
template<MemSpace M>
void DirectGeneratorAction::step_impl(CoreParams const& params,
                                      CoreState<M>& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state
        = get<DirectGeneratorState<M>>(*state.aux(), this->aux_id());
    auto& counters = aux_state.counters;

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
 * Launch a (host) kernel to initialize optical photons.
 */
void DirectGeneratorAction::generate(CoreParams const& params,
                                     CoreStateHost& state) const
{
    CELER_EXPECT(state.aux());

    auto& aux_state = get<DirectGeneratorState<MemSpace::native>>(
        *state.aux(), this->aux_id());
    size_type num_gen = min(state.sync_get_counters().num_vacancies,
                            aux_state.counters.num_pending);

    // Generate optical photons in vacant track slots
    detail::DirectGeneratorExecutor execute{
        params.ptr<MemSpace::native>(), state.ptr(), aux_state.store.ref()};
    launch_action(num_gen, execute);
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void DirectGeneratorAction::generate(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("device");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
