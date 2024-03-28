//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.cu
//---------------------------------------------------------------------------//
#include "AbsorptionModel.hh"

#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void AbsorptionModel::execute(CoreParams const& params,
                              CoreStateDevice& state) const
{
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
