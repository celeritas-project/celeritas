//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/KleinNishinaModel.cc
//---------------------------------------------------------------------------//
#include "KleinNishinaModel.hh"

#include "corecel/Assert.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/em/generated/KleinNishinaInteract.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
KleinNishinaModel::KleinNishinaModel(ActionId              id,
                                     const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.ids.action   = id;
    interface_.ids.electron = particles.find(pdg::electron());
    interface_.ids.gamma    = particles.find(pdg::gamma());

    CELER_VALIDATE(interface_.ids.electron && interface_.ids.gamma,
                   << "missing electron, positron and/or gamma particles "
                      "(required for "
                   << this->description() << ")");
    interface_.inv_electron_mass
        = 1 / particles.get(interface_.ids.electron).mass().value();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto KleinNishinaModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = interface_.ids.gamma;
    photon_applic.lower    = zero_quantity();
    photon_applic.upper    = max_quantity();

    return {photon_applic};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void KleinNishinaModel::execute(CoreDeviceRef const& data) const
{
    generated::klein_nishina_interact(interface_, data);
}

void KleinNishinaModel::execute(CoreHostRef const& data) const
{
    generated::klein_nishina_interact(interface_, data);
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId KleinNishinaModel::action_id() const
{
    return interface_.ids.action;
}

//!@}
//---------------------------------------------------------------------------//
} // namespace celeritas
