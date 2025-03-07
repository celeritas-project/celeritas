//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionModel.cu
//---------------------------------------------------------------------------//
#include "AbsorptionModel.hh"

#include "celeritas/optical/action/ActionLauncher.device.hh"
#include "celeritas/optical/action/TrackSlotExecutor.hh"

#include "../CoreParams.hh"
#include "../CoreState.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Interact with device data.
 */
void AbsorptionModel::step(CoreParams const&, CoreStateDevice&) const {}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
