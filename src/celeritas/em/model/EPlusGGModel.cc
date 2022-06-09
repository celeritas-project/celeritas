//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/EPlusGGModel.cc
//---------------------------------------------------------------------------//
#include "EPlusGGModel.hh"

#include "corecel/Assert.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/em/generated/EPlusGGInteract.hh"
#include "celeritas/phys/Applicability.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
EPlusGGModel::EPlusGGModel(ActionId id, const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.ids.action   = id;
    interface_.ids.positron = particles.find(pdg::positron());
    interface_.ids.gamma    = particles.find(pdg::gamma());

    CELER_VALIDATE(interface_.ids.positron && interface_.ids.gamma,
                   << "missing positron and/or gamma particles (required for "
                   << this->description() << ")");
    interface_.electron_mass
        = particles.get(interface_.ids.positron).mass().value();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto EPlusGGModel::applicability() const -> SetApplicability
{
    Applicability applic;
    applic.particle = interface_.ids.positron;
    applic.lower    = neg_max_quantity();    // Valid at rest
    applic.upper    = units::MevEnergy{1e8}; // 100 TeV

    return {applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto EPlusGGModel::micro_xs(Applicability) const -> MicroXsBuilders
{
    // Cross sections are calculated on the fly
    MicroXsBuilders builders;
    return builders;
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void EPlusGGModel::execute(CoreDeviceRef const& data) const
{
    generated::eplusgg_interact(interface_, data);
}

void EPlusGGModel::execute(CoreHostRef const& data) const
{
    generated::eplusgg_interact(interface_, data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId EPlusGGModel::action_id() const
{
    return interface_.ids.action;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
