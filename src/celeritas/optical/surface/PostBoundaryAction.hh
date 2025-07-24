//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/PostBoundaryAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/action/ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Finalize an optical boundary crossing action.
 *
 * Handle an optical photon finishing a boundary crossing and leaving a
 * surface. The geometry state is updated depending on if it enters the
 * post-volume or is reentrant on the pre-volume.
 */
class PostBoundaryAction : public OpticalStepActionInterface,
                           public ConcreteAction
{
  public:
    // Construct with ID
    explicit PostBoundaryAction(ActionId);

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
