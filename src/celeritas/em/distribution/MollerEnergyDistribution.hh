//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/MollerEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/random/distribution/RejectionSampler.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c MollerBhabhaInteractor .
 *
 * Sample the exiting energy for Moller scattering.
 */
class MollerEnergyDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with data from MollerBhabhaInteractor
    inline CELER_FUNCTION MollerEnergyDistribution(Mass electron_mass,
                                                   Energy min_valid_energy,
                                                   Energy inc_energy);

    // Sample the exiting energy
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    //// DATA ////

    // Electron incident energy [MeV]
    real_type inc_energy_;
    // Total energy of the incident particle [MeV]
    real_type total_energy_;
    // Minimum energy fraction transferred to free electron [MeV]
    real_type min_energy_fraction_;
    // Sampling parameter
    real_type gamma_;

    //// HELPER FUNCTIONS ////

    // Helper function for calculating rejection function g
    inline CELER_FUNCTION real_type calc_g_fraction(real_type epsilon);
    // Maximum energy fraction transferred to free electron [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type max_energy_fraction()
    {
        return 0.5;
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with data from MollerBhabhaInteractor.
 */
CELER_FUNCTION
MollerEnergyDistribution::MollerEnergyDistribution(Mass electron_mass,
                                                   Energy min_valid_energy,
                                                   Energy inc_energy)
    : inc_energy_(value_as<Energy>(inc_energy))
    , total_energy_(inc_energy_ + value_as<Mass>(electron_mass))
    , min_energy_fraction_(value_as<Energy>(min_valid_energy) / inc_energy_)
    , gamma_(total_energy_ / value_as<Mass>(electron_mass))
{
    CELER_EXPECT(electron_mass > zero_quantity() && inc_energy_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample epsilon for Moller scattering.
 */
template<class Engine>
CELER_FUNCTION real_type MollerEnergyDistribution::operator()(Engine& rng)
{
    real_type const g_denominator
        = this->calc_g_fraction(this->max_energy_fraction());

    UniformRealDistribution<> sample_inverse_epsilon(
        1 / this->max_energy_fraction(), 1 / min_energy_fraction_);

    // Sample epsilon
    real_type epsilon;
    do
    {
        epsilon = 1 / sample_inverse_epsilon(rng);
    } while (RejectionSampler<>(calc_g_fraction(epsilon), g_denominator)(rng));

    return epsilon;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the rejection function g.
 */
CELER_FUNCTION real_type
MollerEnergyDistribution::calc_g_fraction(real_type epsilon)
{
    real_type const two_gamma_term = (2 * gamma_ - 1) / ipow<2>(gamma_);
    real_type const complement_frac = 1 - epsilon;

    return 1 - two_gamma_term * epsilon
           + ipow<2>(epsilon)
                 * (1 - two_gamma_term
                    + (1 - two_gamma_term * complement_frac)
                          / ipow<2>(complement_frac));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
