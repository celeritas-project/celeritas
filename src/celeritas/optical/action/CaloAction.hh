//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/CaloAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include "ActionInterface.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Check track interaction with optical detector, deposit (print) energy and
 * kill track
 */
class CaloAction final : public OpticalStepActionInterface,
                         public StaticConcreteAction
{
  public:
    // Construct with ID
    explicit CaloAction(ActionId);

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;

    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Dependency ordering of the action
    StepActionOrder order() const final { return StepActionOrder::user_post; }
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas