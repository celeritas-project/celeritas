//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
void NeutronInelasticModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_IMPLEMENTED("Neutron inelastic interaction");
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
