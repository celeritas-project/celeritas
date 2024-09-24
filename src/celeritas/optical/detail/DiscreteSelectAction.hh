//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/DiscreteSelectAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/ActionInterface.hh"

#include "../CoreParams.hh"
#include "../CoreState.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Select a model for tracks undergoing a discrete interaction.
 */
class DiscreteSelectAction final
    : public StepActionInterface<CoreParams, CoreState>,
      public ConcreteAction
{
  public:
    //! Construct with ID
    explicit DiscreteSelectAction(ActionId);

    //! Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final override;

    //! Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final override;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::pre_post; }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
