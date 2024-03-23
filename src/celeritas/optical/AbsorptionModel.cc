//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.cc
//---------------------------------------------------------------------------//
#include "AbsorptionModel.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

AbsorptionModel::AbsorptionModel(ActionId id, MaterialParams const&)
    : OpticalModel(id)
{
}

void AbsorptionModel::execute(CoreParams const& params,
                              CoreStateHost& state) const
{
}

#if !CELER_USE_DEVICE
void AbsorptionModel::execute(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
