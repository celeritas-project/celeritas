//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/action/DiscreteSelectAction.cu
//---------------------------------------------------------------------------//
#include "DiscreteSelectAction.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Launch the discrete-select action on device.
 */
void DiscreteSelectAction::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NON_IMPLEMENTED("Optical discrete select executor not implemented.");
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
