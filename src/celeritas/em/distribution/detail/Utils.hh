//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/detail/Utils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/Constants.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/ParticleTrackView.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum energy transferable to a free electron in ionizaation.
 *
 * This calculates the maximum kinematically allowed kinetic energy of the
 * delta ray produced in muon or hadron ionization,
 * \f[
   T_{max} = \frac{2 m_e c^2 (\gamma^2 - 1)}{1 + 2\gamma (m_e/M) + (m_e/M)^2},
 * \f]
 * where \f$ m_e \f$ is the electron mass and \f$ M \f$ is the mass of the
 * incident particle.
 */
inline CELER_FUNCTION units::MevEnergy
calc_max_secondary_energy(ParticleTrackView const& particle,
                          units::MevMass electron_mass)
{
    real_type inc_mass = value_as<units::MevMass>(particle.mass());
    real_type mass_ratio = value_as<units::MevMass>(electron_mass) / inc_mass;
    real_type tau = value_as<units::MevEnergy>(particle.energy()) / inc_mass;
    return units::MevEnergy{
        2 * value_as<units::MevMass>(electron_mass) * tau * (tau + 2)
        / (1 + 2 * (tau + 1) * mass_ratio + ipow<2>(mass_ratio))};
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
