//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/detail/DiscreteSelectAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a model for tracks undergoing a discrete interaction.
 */
class DiscreteSelectAction final : public CoreStepActionInterface,
                                   public ConcreteAction
{
  public:
    // Construct with ID
    explicit DiscreteSelectAction(ActionId);

    // Launch kernel with host data
    void execute(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void execute(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::pre_post; }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

#if !CELER_USE_DEVICE
inline void
DiscreteSelectAction::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
