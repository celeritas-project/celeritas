//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/woodcock/WoodcockAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/sys/ActionInterface.hh"
#include "celeritas/global/ActionInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Move photons to a collision or to the edge of a Woodcock tracking region.
 *
 * This is the "along-step" action for tracks selected during the physics
 * pre-step, but it runs \em before the user's pre-step actions. Since a step
 * can cross any number of volumes, the pre-step point recorded by the user is
 * at the end of the flight, in the volume where the interaction takes place
 * and energy is deposited.
 *
 * \sa detail::WoodcockExecutor
 */
class WoodcockAction final : public CoreStepActionInterface,
                             public StaticConcreteAction
{
  public:
    // Construct with ID
    explicit WoodcockAction(ActionId);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::pre; }
};

//---------------------------------------------------------------------------//
/*!
 * Indicate that all sampled collisions were rejected.
 *
 * This implicit action is used when the maximum number of flights per step was
 * reached. The track continues in the next step.
 */
class WoodcockNullAction final : public StaticConcreteAction
{
  public:
    // Construct with ID
    explicit WoodcockNullAction(ActionId);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
