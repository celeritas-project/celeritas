//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/BoundaryAction.cc
//---------------------------------------------------------------------------//
#include "BoundaryAction.hh"

#include "corecel/io/Logger.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Construct with action ID.
 */
BoundaryAction::BoundaryAction(ActionId aid)
    : ConcreteAction(aid, "geo-boundary", "cross a geometry boundary")
{
}

//---------------------------------------------------------------------------//
/*!
 * Launch the boundary action on host.
 */
void BoundaryAction::step(CoreParams const&, CoreStateHost&) const
{
    CELER_LOG_LOCAL(error) << "Boundary action is not implemented";
}

#if !CELER_USE_DEVICE
void BoundaryAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
