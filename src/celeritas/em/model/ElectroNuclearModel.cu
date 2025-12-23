//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/ElectroNuclearModel.cu
//---------------------------------------------------------------------------//
#include "ElectroNuclearModel.hh"

#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interact with device data.
 */
void ElectroNuclearModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("Electro-nuclear interaction");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
