//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGInteractor.i.hh
//---------------------------------------------------------------------------//

#include "base/ArrayUtils.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/GenerateCanonical.hh"
#include "random/distributions/IsotropicDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
EPlusGGInteractor::EPlusGGInteractor(const EPlusGGPointers&     shared,
                                     const ParticleTrackView&   particle,
                                     const Real3&               inc_direction,
                                     StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{
    CELER_EXPECT(particle.particle_id() == shared_.positron_id);
}

//---------------------------------------------------------------------------//
/*!
 * Sample two gammas using the G4eeToTwoGammaModel model.
 *
 * Polarization is *not* implemented.
 */
template<class Engine>
CELER_FUNCTION Interaction EPlusGGInteractor::operator()(Engine& rng)
{
    // Allocate space for two gammas
    Secondary* secondaries = this->allocate_(2);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for two secondaries
        return Interaction::from_failure();
    }

    // Construct an interaction with an absorbed process
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, 2};

    // Sample two gammas
    secondaries[0].particle_id = secondaries[1].particle_id = shared_.gamma_id;

    if (inc_energy_ == 0)
    {
        // Save outgoing secondary data
        secondaries[0].energy = secondaries[1].energy
            = units::MevEnergy{shared_.electron_mass};

        IsotropicDistribution<real_type> gamma_dir;
        secondaries[0].direction = gamma_dir(rng);
        for (int i = 0; i < 3; ++i)
        {
            secondaries[1].direction[i] = -secondaries[0].direction[i];
        }
    }
    else
    {
        constexpr real_type half    = 0.5;
        const real_type     tau     = inc_energy_ / shared_.electron_mass;
        const real_type     tau2    = tau + 2;
        const real_type     sqgrate = std::sqrt(tau / tau2) * half;

        // Evaluate limits of the energy sampling
        const real_type epsilmin     = half - sqgrate;
        const real_type epsilmax     = half + sqgrate;
        const real_type log_epsilqot = std::log(epsilmax / epsilmin);

        // Sample the energy rate of the created gammas
        real_type epsil;
        do
        {
            epsil = epsilmin * std::exp(log_epsilqot * generate_canonical(rng));
        } while (BernoulliDistribution(1 - epsil
                                       + (2 * (tau + 1) * epsil - 1)
                                             / (epsil * tau2 * tau2))(rng));

        // Scattered Gamma angles
        const real_type cost = (epsil * tau2 - 1)
                               / (epsil * std::sqrt(tau * tau2));
        CELER_ASSERT(std::fabs(cost) <= 1);

        // Kinematic of the gamma pair
        const real_type total_energy = inc_energy_ + 2 * shared_.electron_mass;
        const real_type gamma_energy = epsil * total_energy;
        const real_type eplus_moment = std::sqrt(inc_energy_ * total_energy);

        // Sample and save outgoing secondary data
        UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

        secondaries[0].energy = units::MevEnergy{gamma_energy};
        secondaries[0].direction
            = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

        secondaries[1].energy = units::MevEnergy{total_energy - gamma_energy};
        for (int i = 0; i < 3; ++i)
        {
            secondaries[1].direction[i] = eplus_moment * inc_direction_[i]
                                          - inc_energy_ * inc_direction_[i];
        }
        normalize_direction(&secondaries[1].direction);
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
