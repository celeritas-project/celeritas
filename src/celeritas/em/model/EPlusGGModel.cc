//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/EPlusGGModel.cc
//---------------------------------------------------------------------------//
#include "EPlusGGModel.hh"

#include "celeritas/Quantities.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/generated/EPlusGGInteract.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
EPlusGGModel::EPlusGGModel(ActionId id, ParticleParams const& particles)
{
    CELER_EXPECT(id);
    data_.ids.action = id;
    data_.ids.positron = particles.find(pdg::positron());
    data_.ids.gamma = particles.find(pdg::gamma());

    CELER_VALIDATE(data_.ids.positron && data_.ids.gamma,
                   << "missing positron and/or gamma particles (required for "
                   << this->description() << ")");
    data_.electron_mass = particles.get(data_.ids.positron).mass();
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto EPlusGGModel::applicability() const -> SetApplicability
{
    Applicability applic;
    applic.particle = data_.ids.positron;
    applic.lower = neg_max_quantity();  // Valid at rest
    applic.upper = units::MevEnergy{1e8};  // 100 TeV

    return {applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto EPlusGGModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Discrete interaction is material independent, so no element is sampled
    return {};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void EPlusGGModel::execute(CoreParams const& params, CoreStateHost& state) const
{
    generated::eplusgg_interact(params, state, this->host_ref());
}

void EPlusGGModel::execute(CoreParams const& params,
                           CoreStateDevice& state) const
{
    generated::eplusgg_interact(
        params, state, this->device_ref(), this->action_id());
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId EPlusGGModel::action_id() const
{
    return data_.ids.action;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
