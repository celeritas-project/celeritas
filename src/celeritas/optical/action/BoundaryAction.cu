//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/BoundaryAction.cu
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
 * Launch the boundary action on device.
 */
void BoundaryAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_LOG_LOCAL(error) << "Boundary action is not implemented";
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
