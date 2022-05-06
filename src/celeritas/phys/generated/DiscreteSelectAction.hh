//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/generated/DiscreteSelectAction.hh
//! \note Auto-generated by gen-action.py: DO NOT MODIFY!
//---------------------------------------------------------------------------//
#include "celeritas/global/ActionInterface.hh"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "celeritas/global/CoreTrackData.hh"

namespace celeritas
{
namespace generated
{
//---------------------------------------------------------------------------//
class DiscreteSelectAction final : public ExplicitActionInterface, public ConcreteAction
{
public:
  // Construct with ID and label
  using ConcreteAction::ConcreteAction;

  // Launch kernel with host data
  void execute(CoreHostRef const&) const final;

  // Launch kernel with device data
  void execute(CoreDeviceRef const&) const final;
};

#if !CELER_USE_DEVICE
inline void DiscreteSelectAction::execute(CoreDeviceRef const&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace generated
} // namespace celeritas
