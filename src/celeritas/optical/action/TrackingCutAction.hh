//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/TrackingCutAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Kill misbehaving photons and deposit energy locally.
 */
class TrackingCutAction final : public OpticalStepActionInterface,
                                public StaticConcreteAction
{
  public:
    // Construct with ID
    explicit TrackingCutAction(ActionId);

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
