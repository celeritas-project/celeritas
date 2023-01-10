//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepGatherAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/global/ActionInterface.hh"

#include "../StepData.hh"
#include "../StepInterface.hh"
#include "StepStorage.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Gather track step properties at a point during the step.
 *
 * This implementation class is constructed by the StepCollector.
 *
 * TODO: this class is only thread safe by locking it across multiple threads.
 * We'll need thread-independent states *or* a stream ID in the core state
 * corresponding to one element in an array of state data.
 */
template<StepPoint P>
class StepGatherAction final : public ExplicitActionInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPStepStorage = std::shared_ptr<StepStorage>;
    using SPStepInterface = std::shared_ptr<StepInterface>;
    using VecInterface = std::vector<SPStepInterface>;
    //!@}

  public:
    // Construct with action ID and storage
    StepGatherAction(ActionId id, SPStepStorage storage, VecInterface callbacks);

    // Launch kernel with host data
    void execute(CoreHostRef const&) const final;

    // Launch kernel with device data
    void execute(CoreDeviceRef const&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the action
    std::string label() const final
    {
        return P == StepPoint::pre    ? "step-gather-pre"
               : P == StepPoint::post ? "step-gather-post"
                                      : "";
    }

    // Name of the action (for user output)
    std::string description() const final;

    //! Dependency ordering of the action
    ActionOrder order() const final
    {
        return P == StepPoint::pre    ? ActionOrder::pre
               : P == StepPoint::post ? ActionOrder::post_post
                                      : ActionOrder::size_;
    }

  private:
    //// DATA ////

    ActionId id_;
    SPStepStorage storage_;
    VecInterface callbacks_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    inline StepStateData<Ownership::reference, M> const&
    get_state(CoreRef<M> const& core_data) const;
};

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a reference to the step state data, allocating if needed.
 */
template<StepPoint P>
template<MemSpace M>
StepStateData<Ownership::reference, M> const&
StepGatherAction<P>::get_state(CoreRef<M> const& core) const
{
    auto& state_store = storage_->get_state(StepStorage::MemSpaceTag<M>{});
    if (CELER_UNLIKELY(!state_store))
    {
        // State storage hasn't been allocated yet: allocate based on current
        // state
        state_store = CollectionStateStore<StepStateData, M>{
            storage_->params.host_ref(), core.states.size()};
    }
    CELER_ENSURE(state_store);
    return state_store.ref();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
