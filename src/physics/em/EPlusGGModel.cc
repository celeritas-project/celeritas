//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGModel.cc
//---------------------------------------------------------------------------//
#include "EPlusGGModel.hh"

#include "base/Assert.hh"
#include "base/Quantity.hh"
#include "physics/base/Applicability.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/base/ParticleView.hh"
#include "physics/base/Units.hh"
#include "physics/em/detail/EPlusGGData.hh"
#include "physics/em/generated/EPlusGGInteract.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
EPlusGGModel::EPlusGGModel(ModelId id, const ParticleParams& particles)
{
    CELER_EXPECT(id);
    interface_.ids.model    = id;
    interface_.ids.positron = particles.find(pdg::positron());
    interface_.ids.gamma    = particles.find(pdg::gamma());

    CELER_VALIDATE(interface_.ids.positron && interface_.ids.gamma,
                   << "missing positron and/or gamma particles (required for "
                   << this->label() << ")");
    interface_.electron_mass
        = particles.get(interface_.ids.positron).mass().value();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 *
 * \todo The Interactor allows non-rest energies as does the
 * G4eeToTwoGammaModel, which also defines ComputeCrossSectionPerAtom, but
 * does not seem to export cross sections. Is it because they're generated on
 * the fly?
 */
auto EPlusGGModel::applicability() const -> SetApplicability
{
    Applicability in_flight;
    in_flight.particle = interface_.ids.positron;
    in_flight.lower    = zero_quantity();
    in_flight.upper    = units::MevEnergy{1e8}; // 100 TeV

    return {Applicability::at_rest(interface_.ids.positron), in_flight};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void EPlusGGModel::interact(const DeviceInteractRef& data) const
{
    generated::eplusgg_interact(interface_, data);
}

void EPlusGGModel::interact(const HostInteractRef& data) const
{
    generated::eplusgg_interact(interface_, data);
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId EPlusGGModel::model_id() const
{
    return interface_.ids.model;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
