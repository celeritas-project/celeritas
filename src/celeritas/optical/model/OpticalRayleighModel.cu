//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/OpticalRayleighModel.cu
//---------------------------------------------------------------------------//
#include "OpticalRayleighModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Execute model on device.
 */
void OpticalRayleighModel::execute(OpticalParams const&,
                                   OpticalStateDevice&) const
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
