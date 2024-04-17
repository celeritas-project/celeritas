//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/NeutronInelasticModel.cu
//---------------------------------------------------------------------------//
#include "NeutronInelasticModel.hh"

#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interact with device data.
 */
void NeutronInelasticModel::execute(CoreParams const& params,
                                    CoreStateDevice& state) const
{
    CELER_NOT_IMPLEMENTED("Neutron inelastic interaction");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
