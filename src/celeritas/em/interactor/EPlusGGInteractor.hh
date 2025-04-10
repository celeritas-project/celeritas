//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/EPlusGGInteractor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/random/distribution/BernoulliDistribution.hh"
#include "corecel/random/distribution/ReciprocalDistribution.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/EPlusGGData.hh"
#include "celeritas/phys/Interaction.hh"
#include "celeritas/phys/InteractionUtils.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/Secondary.hh"

namespace celeritas
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
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with shared and state data
    inline CELER_FUNCTION
    EPlusGGInteractor(EPlusGGData const& shared,
                      ParticleTrackView const& particle,
                      Real3 const& inc_direction,
                      StackAllocator<Secondary>& allocate);

    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

  private:
    // Shared constant physics properties
    EPlusGGData const& shared_;
    // Incident positron energy [MevEnergy]
    real_type const inc_energy_;
    // Incident direction
    Real3 const& inc_direction_;
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
EPlusGGInteractor::EPlusGGInteractor(EPlusGGData const& shared,
                                     ParticleTrackView const& particle,
                                     Real3 const& inc_direction,
                                     StackAllocator<Secondary>& allocate)
    : shared_(shared)
    , inc_energy_(value_as<Energy>(particle.energy()))
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{
    CELER_EXPECT(particle.particle_id() == shared_.positron);
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
    Secondary* secondaries = allocate_(2);
    if (secondaries == nullptr)
    {
        // Failed to allocate space for two secondaries
        return Interaction::from_failure();
    }

    // Construct an interaction with an absorbed process
    Interaction result = Interaction::from_absorption();
    result.secondaries = {secondaries, 2};

    // Sample two gammas
    secondaries[0].particle_id = secondaries[1].particle_id = shared_.gamma;

    if (inc_energy_ == 0)
    {
        // Save outgoing secondary data
        secondaries[0].energy = secondaries[1].energy
            = Energy{value_as<Mass>(shared_.electron_mass)};

        IsotropicDistribution<real_type> gamma_dir;
        secondaries[0].direction = gamma_dir(rng);
        secondaries[1].direction = -secondaries[0].direction;
    }
    else
    {
        constexpr real_type half = 0.5;
        real_type const tau = inc_energy_
                              / value_as<Mass>(shared_.electron_mass);
        real_type const tau2 = tau + 2;
        real_type const sqgrate = std::sqrt(tau / tau2) * half;

        // Evaluate limits of the energy sampling
        ReciprocalDistribution<real_type> sample_eps(half - sqgrate,
                                                     half + sqgrate);

        // Sample the energy rate of the created gammas
        real_type epsil;
        do
        {
            epsil = sample_eps(rng);
        } while (BernoulliDistribution(
            epsil - (2 * (tau + 1) * epsil - 1) / (epsil * ipow<2>(tau2)))(rng));

        // Scattered Gamma angles
        real_type const cost = (epsil * tau2 - 1)
                               / (epsil * std::sqrt(tau * tau2));
        CELER_ASSERT(std::fabs(cost) <= 1);

        // Kinematic of the gamma pair
        real_type const total_energy
            = inc_energy_ + 2 * value_as<Mass>(shared_.electron_mass);
        real_type const gamma_energy = epsil * total_energy;
        real_type const eplus_moment = std::sqrt(inc_energy_ * total_energy);

        // Sample and save outgoing secondary data
        secondaries[0].energy = Energy{gamma_energy};
        secondaries[0].direction
            = ExitingDirectionSampler{cost, inc_direction_}(rng);
        secondaries[1].energy = Energy{total_energy - gamma_energy};
        secondaries[1].direction = calc_exiting_direction(
            {eplus_moment, inc_direction_}, {inc_energy_, inc_direction_});
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
