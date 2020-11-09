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

#include <iostream>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
EPlusGGInteractor::EPlusGGInteractor(const EPlusGGInteractorPointers& shared,
                                     const ParticleTrackView&         particle,
                                     const Real3&            inc_direction,
                                     SecondaryAllocatorView& allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{
    REQUIRE(inc_energy_ >= this->min_incident_energy()
            && inc_energy_ <= this->max_incident_energy());
    REQUIRE(particle.def_id() == shared_.positron_id);
}

//---------------------------------------------------------------------------//
/*!
 * Sample two gammas using the G4eeToTwoGammaModel model.
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
    secondaries[0].def_id = secondaries[1].def_id = shared_.gamma_id;

    const real_type                  inc_energy = inc_energy_.value();
    IsotropicDistribution<real_type> gamma_dir;

    if (inc_energy == 0)
    {
        // Save outgoing secondary data
        secondaries[0].energy = secondaries[1].energy
            = units::MevEnergy{shared_.electron_mass};

        secondaries[0].direction = gamma_dir(rng);
        for (int i = 0; i < 3; ++i)
        {
            secondaries[1].direction[i] = -secondaries[0].direction[i];
        }

        // Note: polarization is intentionally not implemented
    }
    else
    {
        real_type tau     = inc_energy / shared_.electron_mass;
        real_type gam     = tau + 1.0;
        real_type tau2    = tau + 2.0;
        real_type sqgrate = std::sqrt(tau / tau2) * 0.5;
        real_type sqg2m1  = std::sqrt(tau * tau2);

        // Evaluate limits of the energy sampling
        real_type epsilmin = 0.5 - sqgrate;
        real_type epsilmax = 0.5 + sqgrate;
        real_type epsilqot = epsilmax / epsilmin;

        // Sample the energy rate of the created gammas
        real_type epsil;
        do
        {
            epsil = epsilmin
                    * std::exp(std::log(epsilqot) * generate_canonical(rng));
        } while (BernoulliDistribution(
            1. - epsil + (2. * gam * epsil - 1.) / (epsil * tau2 * tau2))(rng));

        // Scattered Gamma angles
        real_type cost = (epsil * tau2 - 1.0) / (epsil * sqg2m1);
        CHECK(fabs(cost) <= 1.0);

        // Kinematic of the gamma pair
        real_type total_energy = inc_energy + 2.0 * shared_.electron_mass;
        real_type gamma_energy = epsil * total_energy;
        real_type eplus_moment = std::sqrt(inc_energy * total_energy);

        // Sample and save outgoing secondary data
        UniformRealDistribution<real_type> sample_phi(0, 2 * constants::pi);

        secondaries[0].energy = units::MevEnergy{gamma_energy};
        secondaries[0].direction
            = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

        secondaries[1].energy = units::MevEnergy{total_energy - gamma_energy};
        for (int i = 0; i < 3; ++i)
        {
            secondaries[1].direction[i] = eplus_moment * inc_direction_[i]
                                          - inc_energy * inc_direction_[i];
        }
        normalize_direction(&secondaries[1].direction);
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
