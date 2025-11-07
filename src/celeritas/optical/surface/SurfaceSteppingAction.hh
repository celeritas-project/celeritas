//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/SurfaceSteppingAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/action/ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Stepping action for surface physics models during a boundary crossing.
 *
 * Encapsulates all of the surface physics models into a single action, which
 * can be extended to run multiple surface interactions in a single step.
 */
class SurfaceSteppingAction : public OpticalStepActionInterface,
                              public ConcreteAction
{
  public:
    // Construct from action ID
    explicit SurfaceSteppingAction(ActionId);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::post; }

  private:
    // Launch kernel in the given MemSpace
    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
