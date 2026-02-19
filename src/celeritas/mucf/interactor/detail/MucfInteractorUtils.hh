//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mucf/interactor/detail/MucfInteractorUtils.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/IsotropicDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate momentum magnitude from particle energy and mass via
 * \f$ p = \sqrt{K^2 + 2mK} \f$ .
 */
inline CELER_FUNCTION real_type calc_momentum(units::MevEnergy const energy,
                                              units::MevMass const mass)
{
    return std::sqrt(ipow<2>(value_as<units::MevEnergy>(energy))
                     + 2 * value_as<units::MevMass>(mass)
                           * value_as<units::MevEnergy>(energy));
}

//---------------------------------------------------------------------------//
/*!
 * Calculate kinetic energy given a particle's momentum and mass via
 * \f$ K = \sqrt{p^2 - m^2} - m\f$ .
 */
inline CELER_FUNCTION units::MevEnergy
calc_kinetic_energy(Real3 const& momentum_vec, units::MevMass const mass)
{
    real_type momentum_mag = norm(momentum_vec);
    return units::MevEnergy{std::sqrt(ipow<2>(momentum_mag)
                                      + ipow<2>(value_as<units::MevMass>(mass)))
                            - value_as<units::MevMass>(mass)};
}

//---------------------------------------------------------------------------//
/*!
 * Sample muCF secondary with isotropic random direction and known outgoing
 * energy.
 */
template<class Engine>
inline CELER_FUNCTION Secondary sample_mucf_secondary(ParticleId pid,
                                                      units::MevEnergy energy,
                                                      Engine& rng)
{
    Secondary result;
    result.particle_id = pid;
    result.energy = energy;
    result.direction = IsotropicDistribution()(rng);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Sample a muon secondary with isotropic random direction and energy from the
 * provided CDF.
 *
 * \note The muon grid range is [0,1) and its domain is the energy in MeV.
 */
template<class Engine>
inline CELER_FUNCTION Secondary sample_mucf_muon(
    ParticleId pid, NonuniformGridCalculator sample_energy, Engine& rng)
{
    Secondary result;
    result.particle_id = pid;
    result.energy = units::MevEnergy{sample_energy(generate_canonical(rng))};
    result.direction = IsotropicDistribution()(rng);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return opposite direction.
 */
inline CELER_FUNCTION Real3 opposite(Real3 const& vec)
{
    Real3 result;
    for (auto i : range(3))
    {
        result[i] = -vec[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return the momentum of the third secondary from an at-rest three-body
 * sampling, once the other two are known.
 *
 * \todo This may be expanded to do a full three-body energy + momentum
 * conservation.
 */
inline CELER_FUNCTION Secondary
calc_third_secondary(Secondary sec_a,
                     units::MevMass const mass_a,
                     Secondary sec_b,
                     units::MevMass const mass_b,
                     ParticleId pid_c,
                     units::MevMass const mass_c)
{
    auto const momentum_a_mag = calc_momentum(sec_a.energy, mass_a);
    auto const momentum_b_mag = calc_momentum(sec_b.energy, mass_b);

    // Calculate direction from momentum conservation: p3 = - (p1 + p2)
    Real3 momentum_vec;
    for (auto i : range(3))
    {
        momentum_vec[i] = -(sec_a.direction[i] * momentum_a_mag
                            + sec_b.direction[i] * momentum_b_mag);
    }

    Secondary result;
    result.particle_id = pid_c;
    result.direction = make_unit_vector(momentum_vec);
    result.energy = calc_kinetic_energy(momentum_vec, mass_c);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
