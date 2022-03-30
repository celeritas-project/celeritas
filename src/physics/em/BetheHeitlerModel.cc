//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerModel.cc
//---------------------------------------------------------------------------//
#include "BetheHeitlerModel.hh"

#include "base/Assert.hh"
#include "physics/base/PDGNumber.hh"
#include "physics/em/generated/BetheHeitlerInteract.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
BetheHeitlerModel::BetheHeitlerModel(ModelId               id,
                                     const ParticleParams& particles,
                                     bool                  enable_lpm)
{
    CELER_EXPECT(id);
    interface_.ids.model    = id;
    interface_.ids.electron = particles.find(pdg::electron());
    interface_.ids.positron = particles.find(pdg::positron());
    interface_.ids.gamma    = particles.find(pdg::gamma());
    interface_.enable_lpm  = enable_lpm;

    CELER_VALIDATE(interface_.ids,
                   << "missing electron, positron and/or gamma particles "
                      "(required for "
                   << this->label() << ")");
    interface_.electron_mass
        = particles.get(interface_.ids.electron).mass().value();
    CELER_ENSURE(interface_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto BetheHeitlerModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = interface_.ids.gamma;
    photon_applic.lower    = zero_quantity();
    photon_applic.upper    = units::MevEnergy{1e8};

    return {photon_applic};
}

//---------------------------------------------------------------------------//
//!@{
/*!
 * Apply the interaction kernel.
 */
void BetheHeitlerModel::interact(const DeviceInteractRef& data) const
{
    generated::bethe_heitler_interact(interface_, data);
}

void BetheHeitlerModel::interact(const HostInteractRef& data) const
{
    generated::bethe_heitler_interact(interface_, data);
}
//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ModelId BetheHeitlerModel::model_id() const
{
    return interface_.ids.model;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
