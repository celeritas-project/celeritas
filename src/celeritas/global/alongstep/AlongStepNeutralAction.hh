//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/alongstep/AlongStepNeutralAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionInterface.hh"

#include "celeritas_config.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Along-step kernel for particles without fields or energy loss.
 *
 * This should only be used for testing and demonstration purposes because real
 * EM physics always has continuous energy loss for charged particles.
 */
class AlongStepNeutralAction final : public ExplicitActionInterface
{
  public:
    // Construct with next action ID
    explicit AlongStepNeutralAction(ActionId id);

    // Launch kernel with host data
    void execute(CoreHostRef const&) const final;

    // Launch kernel with device data
    void execute(CoreDeviceRef const&) const final;

    //! ID of the model
    ActionId action_id() const final { return id_; }

    //! Short name for the along-step kernel
    std::string label() const final { return "along-step-neutral"; }

    //! Name of the model, for user interaction
    std::string description() const final
    {
        return "along-step for neutral particles";
    }

    //! Dependency ordering of the action
    ActionOrder order() const final { return ActionOrder::along; }

  private:
    ActionId id_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void AlongStepNeutralAction::execute(CoreDeviceRef const&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas
