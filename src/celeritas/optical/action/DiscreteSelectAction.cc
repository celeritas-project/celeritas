//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DiscreteSelectAction.cc
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an action ID.
 */
DiscreteSelectAction::DiscreteSelectAction(ActionId id)
    : StaticConcreteAction(id,
                           "optical-discrete-select",
                           "select a discrete optical interaction")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the discrete-select action on host.
 */
void DiscreteSelectAction::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("Optical discrete select executor not implemented.");
}

//---------------------------------------------------------------------------//
/*!
 * Launch the discrete-select action on device.
 */
#if !CELER_USE_DEVICE
void DiscreteSelectAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
