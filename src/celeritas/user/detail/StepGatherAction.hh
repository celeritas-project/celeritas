//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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

#include "StepStorage.hh"
#include "../StepData.hh"
#include "../StepInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Gather track step properties at a point during the step.
 *
 * This implementation class is constructed by the StepCollector.
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
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

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
};

//---------------------------------------------------------------------------//
template<StepPoint P>
void step_gather_device(DeviceCRef<CoreParamsData> const&,
                        DeviceRef<CoreStateData>&,
                        DeviceCRef<StepParamsData> const&,
                        DeviceRef<StepStateData>&);

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
