//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/generated/BoundaryAction.hh
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/CoreTrackData.hh"

namespace celeritas
{
namespace generated
{
//---------------------------------------------------------------------------//
class BoundaryAction final : public ExplicitActionInterface, public ConcreteAction
{
public:
  // Construct with ID and label
  using ConcreteAction::ConcreteAction;

  // Launch kernel with host data
  void execute(CoreParams const&, StateHostRef&) const final;

  // Launch kernel with device data
  void execute(CoreParams const&, StateDeviceRef&) const final;

  //! Dependency ordering of the action
  ActionOrder order() const final { return ActionOrder::post; }
};

#if !CELER_USE_DEVICE
inline void BoundaryAction::execute(CoreParams const&, StateDeviceRef&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace generated
}  // namespace celeritas
