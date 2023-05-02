//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/KleinNishinaModel.cc
//---------------------------------------------------------------------------//
#include "KleinNishinaModel.hh"

#include "corecel/math/Quantity.hh"
#include "celeritas/em/generated/KleinNishinaInteract.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
KleinNishinaModel::KleinNishinaModel(ActionId id,
                                     ParticleParams const& particles)
{
    CELER_EXPECT(id);
    data_.ids.action = id;
    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.gamma = particles.find(pdg::gamma());

    CELER_VALIDATE(data_.ids.electron && data_.ids.gamma,
                   << "missing electron, positron and/or gamma particles "
                      "(required for "
                   << this->description() << ")");
    data_.inv_electron_mass = 1
                              / value_as<KleinNishinaData::Mass>(
                                  particles.get(data_.ids.electron).mass());
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto KleinNishinaModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = data_.ids.gamma;
    photon_applic.lower = zero_quantity();
    photon_applic.upper = max_quantity();

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto KleinNishinaModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Discrete interaction is material independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void KleinNishinaModel::execute(CoreParams const& params,
                                CoreStateHost& state) const
{
    generated::klein_nishina_interact(this->host_ref(), params, state);
}

void KleinNishinaModel::execute(CoreParams const& params,
                                CoreStateDevice& state) const
{
    generated::klein_nishina_interact(this->device_ref(), params, state);
}

//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId KleinNishinaModel::action_id() const
{
    return data_.ids.action;
}

//!@}
//---------------------------------------------------------------------------//
}  // namespace celeritas
