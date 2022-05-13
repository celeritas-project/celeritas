//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MollerBhabhaModel.cc
//---------------------------------------------------------------------------//
#include "MollerBhabhaModel.hh"

#include "corecel/Assert.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/MollerBhabhaData.hh"
#include "celeritas/em/generated/MollerBhabhaInteract.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MollerBhabhaModel::MollerBhabhaModel(ActionId              id,
                                     const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.ids.action   = id;
    interface_.ids.electron = particles.find(pdg::electron());
    interface_.ids.positron = particles.find(pdg::positron());

    CELER_VALIDATE(interface_.ids.electron && interface_.ids.positron,
                   << "missing electron and/or positron particles "
                      "(required for "
                   << this->description() << ")");

    interface_.electron_mass_c_sq
        = particles.get(interface_.ids.electron).mass().value(); // [MeV]

    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MollerBhabhaModel::applicability() const -> SetApplicability
{
    Applicability electron_applic;
    electron_applic.particle = interface_.ids.electron;

    Applicability positron_applic;
    positron_applic.particle = interface_.ids.positron;

    return {electron_applic, positron_applic};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void MollerBhabhaModel::execute(CoreDeviceRef const& data) const
{
    generated::moller_bhabha_interact(interface_, data);
}

void MollerBhabhaModel::execute(CoreHostRef const& data) const
{
    generated::moller_bhabha_interact(interface_, data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId MollerBhabhaModel::action_id() const
{
    return interface_.ids.action;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
