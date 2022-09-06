//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/MuBremsstrahlungModel.cc
//---------------------------------------------------------------------------//
#include "MuBremsstrahlungModel.hh"

#include "corecel/Assert.hh"
#include "celeritas/em/generated/MuBremsstrahlungInteract.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
MuBremsstrahlungModel::MuBremsstrahlungModel(ActionId              id,
                                             const ParticleParams& particles,
                                             SPConstImported       data)
    : imported_(data,
                particles,
                ImportProcessClass::mu_brems,
                ImportModelClass::mu_brems,
                {pdg::mu_minus(), pdg::mu_plus()})
{
    CELER_EXPECT(id);
    interface_.ids.action   = id;
    interface_.ids.gamma    = particles.find(pdg::gamma());
    interface_.ids.mu_minus = particles.find(pdg::mu_minus());
    interface_.ids.mu_plus  = particles.find(pdg::mu_plus());

    CELER_VALIDATE(interface_.ids.gamma && interface_.ids.mu_minus
                       && interface_.ids.mu_plus,
                   << "missing muon and/or gamma particles (required for "
                   << this->description() << ")");

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

    mu_minus_applic.particle = interface_.ids.mu_minus;
    mu_minus_applic.lower    = zero_quantity();
    mu_minus_applic.upper    = interface_.max_incident_energy();

    mu_plus_applic.particle = interface_.ids.mu_plus;
    mu_plus_applic.lower    = mu_minus_applic.lower;
    mu_plus_applic.upper    = mu_minus_applic.upper;

    return {mu_minus_applic, mu_plus_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto MuBremsstrahlungModel::micro_xs(Applicability applic) const
    -> MicroXsBuilders
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void MuBremsstrahlungModel::execute(CoreDeviceRef const& data) const
{
    generated::mu_bremsstrahlung_interact(interface_, data);
}

void MuBremsstrahlungModel::execute(CoreHostRef const& data) const
{
    generated::mu_bremsstrahlung_interact(interface_, data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId MuBremsstrahlungModel::action_id() const
{
    return interface_.ids.action;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
