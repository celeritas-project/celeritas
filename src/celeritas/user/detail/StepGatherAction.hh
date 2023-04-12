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
    void execute(ParamsHostCRef const&, StateHostRef&) const final;

    // Launch kernel with device data
    void execute(ParamsDeviceCRef const&, StateDeviceRef&) const final;

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
    inline StepStateData<Ownership::reference, M>&
    get_state(CoreParamsData<Ownership::const_reference, M> const&,
              CoreStateData<Ownership::reference, M>&) const;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Get a reference to the stream-local step state data, allocating if needed.
template<MemSpace M>
StepStateData<Ownership::reference, M>&
get_stream_state(CoreParamsData<Ownership::const_reference, M> const& params,
                 CoreStateData<Ownership::reference, M>& states,
                 StepStorage* storage);

//---------------------------------------------------------------------------//
// PRIVATE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a reference to the step state data, allocating if needed.
 */
template<StepPoint P>
template<MemSpace M>
StepStateData<Ownership::reference, M>& StepGatherAction<P>::get_state(
    CoreParamsData<Ownership::const_reference, M> const& params,
    CoreStateData<Ownership::reference, M>& states) const
{
    return get_stream_state(params, states, storage_.get());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
