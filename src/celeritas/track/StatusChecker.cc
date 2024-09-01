//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/StatusChecker.cc
//---------------------------------------------------------------------------//
#include "StatusChecker.hh"

#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/Copier.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionLauncher.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/global/TrackExecutor.hh"

#include "detail/StatusCheckExecutor.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Copy from the given collection to host.
 */
template<class T, Ownership W, MemSpace M, class I>
void copy_in_memspace(Collection<T, W, M, I> const& src,
                      Collection<T, W, M, I>* dst)
{
    CELER_EXPECT(src.size() == dst->size());
    Copier<T, M> copy_to_result{(*dst)[AllItems<T, M>{}]};
    copy_to_result(M, src[AllItems<T, M>{}]);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with action and aux IDs.
 */
StatusChecker::StatusChecker(ActionId action_id, AuxId aux_id)
    : action_id_{action_id}, aux_id_{aux_id}
{
    CELER_EXPECT(action_id_);
    CELER_EXPECT(aux_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Description of the action.
 */
std::string_view StatusChecker::description() const
{
    return "verify simulation state";
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto StatusChecker::create_state(MemSpace m,
                                 StreamId id,
                                 size_type size) const -> UPState
{
    return make_aux_state<StatusCheckStateData>(*this, m, id, size);
}

//---------------------------------------------------------------------------//
/*!
 * Set host data at the beginning of a run.
 */
void StatusChecker::begin_run(CoreParams const& params, CoreStateHost&)
{
    return this->begin_run_impl(params);
}

//---------------------------------------------------------------------------//
/*!
 * Set device data at the beginning of a run.
 */
void StatusChecker::begin_run(CoreParams const& params, CoreStateDevice&)
{
    return this->begin_run_impl(params);
}

//---------------------------------------------------------------------------//
/*!
 * Execute with with the last action's ID and the state.
 *
 * This must be called after both \c create_state and \c begin_run .
 */
template<MemSpace M>
void StatusChecker::execute(ActionId prev_action,
                            CoreParams const& params,
                            CoreState<M>& state) const
{
    CELER_EXPECT(data_);
    CELER_EXPECT(prev_action);
    CELER_EXPECT(params.action_reg()->num_actions()
                 == this->ref<M>().orders.size());

    using StateT = AuxStateData<StatusCheckStateData, M>;
    auto& aux_state = get<StateT>(state.aux(), aux_id_).ref();

    // Update action before launching kernel
    CELER_ASSERT(prev_action < this->host_ref().orders.size());
    aux_state.action = prev_action;
    aux_state.order = this->host_ref().orders[prev_action];

    CELER_ASSERT(aux_state.order != StepActionOrder::size_);
    this->launch_impl(params, state, aux_state);

    // Save the status and limiting action IDs
    auto const& sim_state = state.ref().sim;
    copy_in_memspace(sim_state.status, &aux_state.status);
    copy_in_memspace(sim_state.post_step_action, &aux_state.post_step_action);
    copy_in_memspace(sim_state.along_step_action, &aux_state.along_step_action);
}

//---------------------------------------------------------------------------//
/*!
 * Construct host/device data.
 */
void StatusChecker::begin_run_impl(CoreParams const& params)
{
    auto const& reg = *params.action_reg();

    HostVal<StatusCheckParamsData> host_val;
    auto build_orders = CollectionBuilder{&host_val.orders};

    // Loop over all action IDs
    for (auto aidx : range(reg.num_actions()))
    {
        // Get abstract action shared pointer and see if it's explicit
        auto const& base = reg.action(ActionId{aidx});
        if (auto const* expl
            = dynamic_cast<CoreStepActionInterface const*>(base.get()))
        {
            build_orders.push_back(expl->order());
        }
        else
        {
            build_orders.push_back(host_val.implicit_order);
        }
    }
    CELER_ASSERT(build_orders.size() == reg.num_actions());

    // Construct host/device data
    data_ = CollectionMirror{std::move(host_val)};

    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Execute with with the last action's ID and the state.
 */
void StatusChecker::launch_impl(
    CoreParams const& params,
    CoreState<MemSpace::host>& state,
    StatusStateRef<MemSpace::host> const& aux_state) const
{
    launch_core(this->label(),
                params,
                state,
                TrackExecutor{params.ptr<MemSpace::native>(),
                              state.ptr(),
                              detail::StatusCheckExecutor{
                                  this->ref<MemSpace::native>(), aux_state}});
}

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
void StatusChecker::launch_impl(CoreParams const&,
                                CoreState<MemSpace::device>&,
                                StatusStateRef<MemSpace::device> const&) const

{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template void StatusChecker::execute(ActionId,
                                     CoreParams const&,
                                     CoreState<MemSpace::host>&) const;
template void StatusChecker::execute(ActionId,
                                     CoreParams const&,
                                     CoreState<MemSpace::device>&) const;

//---------------------------------------------------------------------------//
}  // namespace celeritas
