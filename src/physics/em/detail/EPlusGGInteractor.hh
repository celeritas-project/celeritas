//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EPlusGGInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/ArrayUtils.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "base/StackAllocator.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "random/distributions/BernoulliDistribution.hh"
#include "random/distributions/IsotropicDistribution.hh"
#include "random/distributions/ReciprocalDistribution.hh"
#include "EPlusGGData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Annihilate a positron to create two gammas.
 *
 * This is a model for the discrete positron-electron annihilation process
 * which simulates the in-flight annihilation of a positron with an atomic
 * electron and produces into two photons. It is assumed that the atomic
 * electron is initially free and at rest.
 *
 * \note This performs the same sampling routine as in Geant4's
 * G4eeToTwoGammaModel class, as documented in section 10.3 of the Geant4
 * Physics Reference (release 10.6). The maximum energy for the model is
 * specified in Geant4.
 */
class EPlusGGInteractor
{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    EPlusGGInteractor(const EPlusGGData&         shared,
                      const ParticleTrackView&   particle,
                      const Real3&               inc_direction,
                      StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    const EPlusGGData& shared_;
    // Incident positron energy [MevEnergy]
    const real_type inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for secondary particles (two photons)
    StackAllocator<Secondary>& allocate_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
EPlusGGInteractor::EPlusGGInteractor(const EPlusGGData&         shared,
                                     const ParticleTrackView&   particle,
                                     const Real3&               inc_direction,
                                     StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_(value_as<units::MevEnergy>(particle.energy()))
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
    // Allocate space for one of the photons (the other is preallocated in the
    // interaction)
    Secondary* gamma2 = this->allocate_(1);
    if (gamma2 == nullptr)
    {
        // Failed to allocate space for two secondaries
        return Interaction::from_failure();
    }

    // Construct an interaction with an absorbed process
    Interaction result = Interaction::from_absorption();
    result.secondaries = {gamma2, 1};
    Secondary* gamma1  = &result.secondary;

    // Sample two gammas
    gamma1->particle_id = gamma2->particle_id = shared_.gamma_id;

    if (inc_energy_ == 0)
    {
        // Save outgoing secondary data
        gamma1->energy = gamma2->energy
            = units::MevEnergy{shared_.electron_mass};

        IsotropicDistribution<real_type> gamma_dir;
        gamma1->direction = gamma_dir(rng);
        for (int i = 0; i < 3; ++i)
        {
            gamma2->direction[i] = -gamma1->direction[i];
        }
    }
    else
    {
        constexpr real_type half    = 0.5;
        const real_type     tau     = inc_energy_ / shared_.electron_mass;
        const real_type     tau2    = tau + 2;
        const real_type     sqgrate = std::sqrt(tau / tau2) * half;

        // Evaluate limits of the energy sampling
        ReciprocalDistribution<real_type> sample_eps(half - sqgrate,
                                                     half + sqgrate);

        // Sample the energy rate of the created gammas
        real_type epsil;
        do
        {
            epsil = sample_eps(rng);
        } while (BernoulliDistribution(1 - epsil
                                       + (2 * (tau + 1) * epsil - 1)
                                             / (epsil * ipow<2>(tau2)))(rng));

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

        gamma1->energy = units::MevEnergy{gamma_energy};
        gamma1->direction
            = rotate(from_spherical(cost, sample_phi(rng)), inc_direction_);

        gamma2->energy = units::MevEnergy{total_energy - gamma_energy};
        for (int i = 0; i < 3; ++i)
        {
            gamma2->direction[i] = eplus_moment * inc_direction_[i]
                                   - inc_energy_ * inc_direction_[i];
        }
        normalize_direction(&gamma2->direction);
    }

    return result;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
