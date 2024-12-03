//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/alongstep/AlongStepNeutralAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>

#include "corecel/Assert.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Along-step kernel for particles without fields or energy loss.
 *
 * This should only be used for testing and demonstration purposes because real
 * EM physics always has continuous energy loss for charged particles.
 */
class AlongStepNeutralAction final : public CoreStepActionInterface
{
  public:
    // Construct with next action ID
    explicit AlongStepNeutralAction(ActionId id);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the along-step kernel
    std::string_view label() const final { return "along-step-neutral"; }

    //! Short description of the action
    std::string_view description() const final
    {
        return "apply along-step for neutral particles";
    }

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::along; }

  private:
    ActionId id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
