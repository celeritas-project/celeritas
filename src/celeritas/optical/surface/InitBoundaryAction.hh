//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/InitBoundaryAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/action/ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Initialize an optical boundary crossing action.
 *
 * Optical surface physics may take many iterations to cross a boundary,
 * depending on its roughness and number of surface layers. This action
 * moves the track across a boundary, calculates the surface normal, and
 * initializes the state of the boundary crossing loop.
 */
class InitBoundaryAction : public OpticalStepActionInterface,
                           public ConcreteAction
{
  public:
    // Construct with ID
    explicit InitBoundaryAction(ActionId);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
