//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/ExtendFromPrimariesAction.cc
//---------------------------------------------------------------------------//
#include "ExtendFromPrimariesAction.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/Copier.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "corecel/sys/MultiExceptionHandler.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/track/TrackInitParams.hh"

#include "detail/ProcessPrimariesExecutor.hh"  // IWYU pragma: associated

namespace celeritas
{
static char const efp_label[] = "extend-from-primaries";

//---------------------------------------------------------------------------//
/*!
 * Construct and add to core params.
 */
std::shared_ptr<ExtendFromPrimariesAction>
ExtendFromPrimariesAction::make_and_insert(CoreParams const& core)
{
    ActionRegistry& actions = *core.action_reg();
    AuxParamsRegistry& aux = *core.aux_reg();
    auto result = std::make_shared<ExtendFromPrimariesAction>(
        actions.next_id(), aux.next_id());
    actions.insert(result);
    aux.insert(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Hacky helper function to get the primary action from core params.
 *
 * This is intended only as a transitional helper function and SHOULD NOT BE
 * USED.
 *
 * \return action if
 */
std::shared_ptr<ExtendFromPrimariesAction const>
ExtendFromPrimariesAction::find_action(CoreParams const& core)
{
    if (auto aid = core.action_reg()->find_action(efp_label))
    {
        auto action
            = std::dynamic_pointer_cast<ExtendFromPrimariesAction const>(
                core.action_reg()->action(aid));
        CELER_VALIDATE(action,
                       << "incorrect type for '" << efp_label << "' action");
        return action;
    }
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Construct with ids.
 */
ExtendFromPrimariesAction::ExtendFromPrimariesAction(ActionId action_id,
                                                     AuxId aux_id)
    : id_{action_id}, aux_id_{aux_id}
{
    CELER_EXPECT(id_ && aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Short name for the action.
 */
std::string_view ExtendFromPrimariesAction::label() const
{
    return efp_label;
}

//---------------------------------------------------------------------------//
/*!
 * Description of the action for user interaction.
 */
std::string_view ExtendFromPrimariesAction::description() const
{
    return "create track initializers from primaries";
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto ExtendFromPrimariesAction::create_state(MemSpace m,
                                             StreamId,
                                             size_type) const -> UPState
{
    if (m == MemSpace::host)
    {
        return std::make_unique<PrimaryStateData<MemSpace::host>>();
    }
    else if (m == MemSpace::device)
    {
        return std::make_unique<PrimaryStateData<MemSpace::device>>();
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Add user-provided primaries on host.
 */
void ExtendFromPrimariesAction::insert(CoreParams const& params,
                                       CoreStateInterface& state,
                                       Span<Primary const> host_primaries) const
{
    size_type num_initializers = state.counters().num_initializers;
    size_type init_capacity = params.init()->capacity();

    CELER_VALIDATE(host_primaries.size() + num_initializers <= init_capacity,
                   << "insufficient initializer capacity (" << init_capacity
                   << ") with size (" << num_initializers
                   << ") for primaries (" << host_primaries.size() << ")");

    if (auto* s = dynamic_cast<CoreState<MemSpace::host>*>(&state))
    {
        this->insert_impl(*s, host_primaries);
    }
    else if (auto* s = dynamic_cast<CoreState<MemSpace::device>*>(&state))
    {
        this->insert_impl(*s, host_primaries);
    }
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with host data.
 */
void ExtendFromPrimariesAction::step(CoreParams const& params,
                                     CoreStateHost& state) const
{
    return this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Execute the action with device data.
 */
void ExtendFromPrimariesAction::step(CoreParams const& params,
                                     CoreStateDevice& state) const
{
    return this->step_impl(params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Add primaries to the core state.
 */
template<MemSpace M>
void ExtendFromPrimariesAction::insert_impl(
    CoreState<M>& state, Span<Primary const> host_primaries) const
{
    auto& pstate = get<PrimaryStateData<M>>(state.aux(), aux_id_);
    if (pstate.count != 0)
    {
        /*!
         * \todo Add support for having more primaries than track initializers
         * or multiple consecutive calls to \c insert.
         */
        CELER_NOT_IMPLEMENTED("multiple consecutive primary insertions");
    }

    if (pstate.storage.size() < host_primaries.size())
    {
        // Reallocate with enough capacity
        resize(&pstate.storage, host_primaries.size());
    }
    pstate.count = host_primaries.size();

    Copier<Primary, M> copy_to_temp{pstate.primaries(), state.stream_id()};
    copy_to_temp(MemSpace::host, host_primaries);
}

//---------------------------------------------------------------------------//
/*!
 * Construct primaries.
 */
template<MemSpace M>
void ExtendFromPrimariesAction::step_impl(CoreParams const& params,
                                          CoreState<M>& state) const
{
    auto& primaries = get<PrimaryStateData<M>>(state.aux(), aux_id_);

    // Create track initializers from primaries
    state.counters().num_initializers += primaries.count;
    this->process_primaries(params, state, primaries);

    // Mark that the primaries have been processed
    state.counters().num_generated += primaries.count;
    primaries.count = 0;

    // Clear the track slot IDs of the track initializers' parent tracks. This
    // is necessary when new primaries are inserted in the middle of a
    // simulation and the parent IDs of secondaries produced in the previous
    // step have been stored.
    fill(TrackSlotId{}, &state.ref().init.parents);
}

//---------------------------------------------------------------------------//
/*!
 * Launch a (host) kernel to create track initializers from primary particles.
 *
 * This function loops over *primaries and initializers*, not *track slots*, so
 * we do not use \c launch_core or \c launch_action .
 */
void ExtendFromPrimariesAction::process_primaries(
    CoreParams const&,
    CoreStateHost& state,
    PrimaryStateData<MemSpace::host> const& pstate) const
{
    MultiExceptionHandler capture_exception;
    auto primaries = pstate.primaries();
    detail::ProcessPrimariesExecutor execute_thread{
        state.ptr(), primaries, state.counters()};
#if defined(_OPENMP) && CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
    for (size_type i = 0, size = primaries.size(); i != size; ++i)
    {
        CELER_TRY_HANDLE(execute_thread(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void ExtendFromPrimariesAction::process_primaries(
    CoreParams const&,
    CoreStateDevice&,
    PrimaryStateData<MemSpace::device> const&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
