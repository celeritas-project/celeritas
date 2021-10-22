//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MuBremsstrahlungModel.cc
//---------------------------------------------------------------------------//
#include "MuBremsstrahlungModel.hh"

#include "base/Assert.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/em/generated/MuBremsstrahlungInteract.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MuBremsstrahlungModel::MuBremsstrahlungModel(ModelId               id,
                                             const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.model_id    = id;
    interface_.gamma_id    = particles.find(pdg::gamma());
    interface_.mu_minus_id = particles.find(pdg::mu_minus());
    interface_.mu_plus_id  = particles.find(pdg::mu_plus());

    CELER_VALIDATE(
        interface_.gamma_id && interface_.mu_minus_id && interface_.mu_plus_id,
        << "missing muon and/or gamma particles "
           "(required for "
        << this->label() << ")");

    interface_.electron_mass
        = particles.get(particles.find(pdg::electron())).mass();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto MuBremsstrahlungModel::applicability() const -> SetApplicability
{
    Applicability mu_minus_applic, mu_plus_applic;

    mu_minus_applic.particle = interface_.mu_minus_id;
    mu_minus_applic.lower    = interface_.min_incident_energy();
    mu_minus_applic.upper    = interface_.max_incident_energy();

    mu_plus_applic.particle = interface_.mu_plus_id;
    mu_plus_applic.lower    = mu_minus_applic.lower;
    mu_plus_applic.upper    = mu_minus_applic.upper;

    return {mu_minus_applic, mu_plus_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Apply the interaction kernel.
 */
void MuBremsstrahlungModel::interact(
    const ModelInteractRef<MemSpace::device>& data) const
{
    generated::mu_bremsstrahlung_interact(interface_, data);
}

void MuBremsstrahlungModel::interact(
    const ModelInteractRef<MemSpace::host>& data) const
{
    generated::mu_bremsstrahlung_interact(interface_, data);
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId MuBremsstrahlungModel::model_id() const
{
    return interface_.model_id;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
