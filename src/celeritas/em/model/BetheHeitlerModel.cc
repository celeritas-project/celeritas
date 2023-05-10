//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/model/BetheHeitlerModel.cc
//---------------------------------------------------------------------------//
#include "BetheHeitlerModel.hh"

#include <utility>

#include "celeritas/Quantities.hh"
#include "celeritas/em/generated/BetheHeitlerInteract.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleView.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from model ID and other necessary data.
 */
BetheHeitlerModel::BetheHeitlerModel(ActionId id,
                                     ParticleParams const& particles,
                                     SPConstImported data,
                                     bool enable_lpm)
    : imported_(data,
                particles,
                ImportProcessClass::conversion,
                ImportModelClass::bethe_heitler_lpm,
                {pdg::gamma()})
{
    CELER_EXPECT(id);
    data_.ids.action = id;
    data_.ids.electron = particles.find(pdg::electron());
    data_.ids.positron = particles.find(pdg::positron());
    data_.ids.gamma = particles.find(pdg::gamma());
    data_.enable_lpm = enable_lpm;

    CELER_VALIDATE(data_.ids,
                   << "missing electron, positron and/or gamma particles "
                      "(required for "
                   << this->description() << ")");
    data_.electron_mass = particles.get(data_.ids.electron).mass();
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * Particle types and energy ranges that this model applies to.
 */
auto BetheHeitlerModel::applicability() const -> SetApplicability
{
    Applicability photon_applic;
    photon_applic.particle = data_.ids.gamma;
    photon_applic.lower = zero_quantity();
    photon_applic.upper = units::MevEnergy{1e8};

    return {photon_applic};
}

//---------------------------------------------------------------------------//
/*!
 * Get the microscopic cross sections for the given particle and material.
 */
auto BetheHeitlerModel::micro_xs(Applicability applic) const -> MicroXsBuilders
{
    return imported_.micro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
//!@{
void BetheHeitlerModel::execute(CoreParams const& params,
                                CoreStateHost& state) const
{
    generated::bethe_heitler_interact(params, state, this->host_ref());
}
/*!
 * Apply the interaction kernel.
 */
void BetheHeitlerModel::execute(CoreParams const& params,
                                CoreStateDevice& state) const
{
    generated::bethe_heitler_interact(params, state, this->device_ref());
}

//!@}
//---------------------------------------------------------------------------//
/*!
 * Get the model ID for this model.
 */
ActionId BetheHeitlerModel::action_id() const
{
    return data_.ids.action;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
