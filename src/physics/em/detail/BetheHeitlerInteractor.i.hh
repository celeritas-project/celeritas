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
 * The incident particle must be above the energy threshold: this should be
 * handled in code *before* the interactor is constructed.
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

    epsilon0_ = 1.0 / (shared_.inv_electron_mass * inc_energy_.value());
    // Gamma energy must be at least 2x electron rest mass
    CELER_ASSERT(epsilon0_ < 0.5);
    static_assert(sizeof(real_type) == sizeof(double),
                  "Embedded constants are hardcoded to double precision.");
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

    // If E_gamma < 2 MeV, rejection not needed -- just sample uniformly
    real_type epsilon;
    if (inc_energy_.value() < 2.0)
    {
        UniformRealDistribution<real_type> sample_eps(epsilon0_, 0.5);
        epsilon = sample_eps(rng);
    }
    else
    {
        // Minimum (\epsilon = 0.5) and maximum (\epsilon = \epsilon1) values
        // of screening variable, \delta.
        real_type delta_min = 136.0 / element_.cbrt_z() * 4.0 * epsilon0_;
        real_type delta_max
            = std::exp((42.24 - element_.coulomb_correction()) / 8.368) - 0.952;
        CELER_ASSERT(delta_min <= delta_max);

        // Limits on epsilon
        real_type epsilon1 = 0.5 - 0.5 * std::sqrt(1.0 - delta_min / delta_max);
        real_type epsilon_min = celeritas::max(epsilon0_, epsilon1);

        // Decide to choose f1, g1 or f2, g2 based on N1, N2 (factors from
        // corrected Bethe-Heitler cross section; c.f. Eq. 6.6 of Geant4
        // Physics Reference 10.6)
        real_type             f10 = this->screening_phi1_aux(delta_min);
        real_type             f20 = this->screening_phi2_aux(delta_min);
        BernoulliDistribution choose_f1g1(ipow<2>(0.5 - epsilon_min) * f10,
                                          1.5 * f20);

        // Temporary sample values used in rejection
        real_type reject_threshold;
        do
        {
            if (choose_f1g1(rng))
            {
                // Used to sample from f1
                epsilon = 0.5
                          - (0.5 - epsilon_min)
                                * std::cbrt(generate_canonical(rng));
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= 0.5);
                // Calculate delta from element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);
                // Calculate g1 "rejection" function
                reject_threshold
                    = celeritas::max(this->screening_phi1_aux(delta), 0.0)
                      / celeritas::max(f10, 0.0);
                CELER_ASSERT(reject_threshold > 0.0 && reject_threshold <= 1.0);
            }
            else
            {
                // Used to sample from f2
                epsilon = epsilon_min
                          + (0.5 - epsilon_min) * generate_canonical(rng);
                CELER_ASSERT(epsilon >= epsilon_min && epsilon <= 0.5);
                // Calculate delta given the element atomic number and sampled
                // epsilon
                real_type delta = this->impact_parameter(epsilon);
                CELER_ASSERT(delta <= delta_max && delta >= delta_min);
                // Calculate g2 "rejection" function
                reject_threshold
                    = celeritas::max(this->screening_phi2_aux(delta), 0.0)
                      / celeritas::max(f20, 0.0);
                CELER_ASSERT(reject_threshold > 0.0 && reject_threshold <= 1.0);
            }
        } while (BernoulliDistribution(1.0 - reject_threshold)(rng));
    }

    // Construct interaction for change to primary (incident) particle (gamma)
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, 2};

    // Outgoing secondaries are electron and positron
    secondaries[0].particle_id = shared_.electron_id;
    secondaries[1].particle_id = shared_.positron_id;
    secondaries[0].energy
        = units::MevEnergy{(1.0 - epsilon) * inc_energy_.value()};
    secondaries[1].energy = units::MevEnergy{epsilon * inc_energy_.value()};
    // Select charges for child particles (e-, e+) randomly
    if (BernoulliDistribution(0.5)(rng))
    {
        swap(secondaries[0].energy, secondaries[1].energy);
    }

    // Sample secondary directions.
    // Note that momentum is not exactly conserved.
    UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);
    real_type                          phi  = sample_phi(rng);
    real_type                          sinp = std::sin(phi);
    real_type                          cosp = std::cos(phi);
    // Electron
    real_type cost = this->sample_cos_theta(secondaries[0].energy.value(), rng);
    secondaries[0].direction
        = rotate(from_spherical(cost, phi), inc_direction_);
    // Positron
    cost           = sample_cos_theta(secondaries[1].energy.value(), rng);
    real_type sint = std::sqrt(1 - ipow<2>(cost));
    secondaries[1].direction
        = rotate({-sint * cosp, -sint * sinp, cost}, inc_direction_);

    return result;
}

template<class Engine>
CELER_FUNCTION real_type
BetheHeitlerInteractor::sample_cos_theta(real_type kinetic_energy, Engine& rng)
{
    real_type umax = 2.0 * (1.0 + kinetic_energy * shared_.inv_electron_mass);
    real_type u;
    do
    {
        real_type uu
            = -std::log(generate_canonical(rng) * generate_canonical(rng));
        u = BernoulliDistribution(0.25)(rng) ? uu * 1.6 : uu * (1.6 / 3.0);
    } while (u > umax);

    return 1.0 - 2.0 * ipow<2>(u / umax);
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::impact_parameter(real_type eps) const
{
    return 136.0 / element_.cbrt_z() * epsilon0_ / (eps * (1.0 - eps));
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi1(real_type delta) const
{
    return delta <= 1.4 ? 20.867 - 3.242 * delta + 0.625 * ipow<2>(delta)
                        : 21.12 - 4.184 * std::log(delta + 0.952);
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi2(real_type delta) const
{
    return delta <= 1.4 ? 20.209 - 1.930 * delta - 0.086 * ipow<2>(delta)
                        : 21.12 - 4.184 * std::log(delta + 0.952);
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi1_aux(real_type delta) const
{
    return (3.0 * this->screening_phi1(delta) - this->screening_phi2(delta)
            - element_.coulomb_correction());
}

CELER_FUNCTION real_type
BetheHeitlerInteractor::screening_phi2_aux(real_type delta) const
{
    return (1.5 * this->screening_phi1(delta)
            - 0.5 * this->screening_phi2(delta)
            - element_.coulomb_correction());
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
