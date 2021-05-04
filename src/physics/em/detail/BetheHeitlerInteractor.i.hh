//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BetheHeitlerInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "base/Constants.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/UniformRealDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 *
 * The incident gamma energy must be at least twice the electron rest mass.
 */
BetheHeitlerInteractor::BetheHeitlerInteractor(
    const BetheHeitlerPointers& shared,
    const ParticleTrackView&    particle,
    const Real3&                inc_direction,
    StackAllocator<Secondary>&  allocate,
    const ElementView&          element)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
    , element_(element)
{
    CELER_EXPECT(particle.particle_id() == shared_.gamma_id);
    CELER_EXPECT(inc_energy_.value() > 2 / shared_.inv_electron_mass);

    epsilon0_ = 1 / (shared_.inv_electron_mass * inc_energy_.value());
    CELER_ENSURE(epsilon0_ < real_type(0.5));
}

//---------------------------------------------------------------------------//
/*!
 * Pair-production using the Bethe-Heitler model.
 *
 * See section 6.5 of the Geant physics reference 10.6.
 */
template<class Engine>
CELER_FUNCTION Interaction BetheHeitlerInteractor::operator()(Engine& rng)
{
    // Allocate space for the pair-produced electrons
    Secondary* secondaries = this->allocate_(2);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }

    constexpr real_type half = 0.5;

    // If E_gamma < 2 MeV, rejection not needed -- just sample uniformly
    real_type epsilon;
    if (inc_energy_ < units::MevEnergy{2.0})
    {
        UniformRealDistribution<real_type> sample_eps(epsilon0_, half);
        epsilon = sample_eps(rng);
    }
    else
    {
        // Minimum (\epsilon = 1/2) and maximum (\epsilon = \epsilon1) values
        // of screening variable, \delta.
        const real_type delta_min = 4 * 136 / element_.cbrt_z() * epsilon0_;
        const real_type delta_max
            = std::exp((real_type(42.24) - element_.coulomb_correction())
                       / real_type(8.368))
              - real_type(0.952);
        CELER_ASSERT(delta_min <= delta_max);

        // Limits on epsilon
        const real_type epsilon1
            = half - half * std::sqrt(1 - delta_min / delta_max);
        const real_type epsilon_min = celeritas::max(epsilon0_, epsilon1);

        // Decide to choose f1, g1 or f2, g2 based on N1, N2 (factors from
        // corrected Bethe-Heitler cross section; c.f. Eq. 6.6 of Geant4
        // Physics Reference 10.6)
        const real_type       f10 = this->screening_phi1_aux(delta_min);
        const real_type       f20 = this->screening_phi2_aux(delta_min);
        BernoulliDistribution choose_f1g1(ipow<2>(half - epsilon_min) * f10,
                                          real_type(1.5) * f20);

        // Temporary sample values used in rejection
        real_type reject_threshold;
        do
        {
            if (choose_f1g1(rng))
            {
                // Used to sample from f1
                epsilon = half
                          - (half - epsilon_min)
                                * std::cbrt(generate_canonical(rng));
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= half);
                // Calculate delta from element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);
                // Calculate g1 "rejection" function
                reject_threshold = this->screening_phi1_aux(delta) / f10;
                CELER_ASSERT(reject_threshold > 0 && reject_threshold <= 1);
            }
            else
            {
                // Used to sample from f2
                epsilon = epsilon_min
                          + (half - epsilon_min) * generate_canonical(rng);
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= half);
                // Calculate delta given the element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);
                // Calculate g2 "rejection" function
                reject_threshold = this->screening_phi2_aux(delta) / f20;
                CELER_ASSERT(reject_threshold > 0 && reject_threshold <= 1);
            }
        } while (!BernoulliDistribution(reject_threshold)(rng));
    }

    // Construct interaction for change to primary (incident) particle (gamma)
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, 2};

    // Outgoing secondaries are electron and positron
    secondaries[0].particle_id = shared_.electron_id;
    secondaries[1].particle_id = shared_.positron_id;
    secondaries[0].energy
        = units::MevEnergy{(1 - epsilon) * inc_energy_.value()};
    secondaries[1].energy = units::MevEnergy{epsilon * inc_energy_.value()};
    // Select charges for child particles (e-, e+) randomly
    if (BernoulliDistribution(half)(rng))
    {
        swap(secondaries[0].energy, secondaries[1].energy);
    }

    // Sample secondary directions.
    // Note that momentum is not exactly conserved.
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    real_type                          phi = sample_phi(rng);

    // Electron
    real_type cost = this->sample_cos_theta(secondaries[0].energy.value(), rng);
    secondaries[0].direction
        = rotate(from_spherical(cost, phi), inc_direction_);
    // Positron
    cost = this->sample_cos_theta(secondaries[1].energy.value(), rng);
    secondaries[1].direction
        = rotate(from_spherical(cost, phi + constants::pi), inc_direction_);
    return result;
}

template<class Engine>
CELER_FUNCTION real_type
BetheHeitlerInteractor::sample_cos_theta(real_type kinetic_energy, Engine& rng)
{
    real_type umax = 2 * (1 + kinetic_energy * shared_.inv_electron_mass);
    real_type u;
    do
    {
        real_type uu
            = -std::log(generate_canonical(rng) * generate_canonical(rng));
        u = uu
            * (BernoulliDistribution(0.25)(rng) ? real_type(1.6)
                                                : real_type(1.6 / 3));
    } while (u > umax);

    return 1 - 2 * ipow<2>(u / umax);
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::impact_parameter(real_type eps) const
{
    return 136 / element_.cbrt_z() * epsilon0_ / (eps * (1 - eps));
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi1(real_type delta) const
{
    using R = real_type;
    return delta <= R(1.4)
               ? R(20.867) - R(3.242) * delta + R(0.625) * ipow<2>(delta)
               : R(21.12) - R(4.184) * std::log(delta + R(0.952));
}

CELER_FUNCTION
real_type BetheHeitlerInteractor::screening_phi2(real_type delta) const
{
    using R = real_type;
    return delta <= R(1.4)
               ? R(20.209) - R(1.930) * delta - R(0.086) * ipow<2>(delta)
               : R(21.12) - R(4.184) * std::log(delta + R(0.952));
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi1_aux(real_type delta) const
{
    // TODO: maybe assert instead of "clamp"? Not clear when this is negative
    return celeritas::clamp_to_nonneg(3 * this->screening_phi1(delta)
                                      - this->screening_phi2(delta)
                                      - element_.coulomb_correction());
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi2_aux(real_type delta) const
{
    // TODO: maybe assert instead of "clamp"? Not clear when this is negative
    using R = real_type;
    return celeritas::clamp_to_nonneg(R(1.5) * this->screening_phi1(delta)
                                      - R(0.5) * this->screening_phi2(delta)
                                      - element_.coulomb_correction());
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
