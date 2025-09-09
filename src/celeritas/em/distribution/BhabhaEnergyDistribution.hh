//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/BhabhaEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/random/distribution/RejectionSampler.hh"
#include "corecel/random/distribution/UniformRealDistribution.hh"
#include "celeritas/Quantities.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c MollerBhabhaInteractor .
 *
 * Sample the exiting energy fraction for Bhabha scattering.
 */
class BhabhaEnergyDistribution
{
  public:
    //!@{
    //! \name Type aliases
    using Mass = units::MevMass;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct with data from MollerBhabhaInteractor
    inline CELER_FUNCTION BhabhaEnergyDistribution(Mass electron_mass,
                                                   Energy min_valid_energy,
                                                   Energy inc_energy);

    // Sample the exiting energy
    template<class Engine>
    inline CELER_FUNCTION real_type operator()(Engine& rng);

  private:
    //// DATA ////

    // Minimum energy fraction transferred to free electron
    real_type min_energy_fraction_;
    // Sampling parameter
    real_type gamma_;

    //// HELPER FUNCTIONS ////

    // Helper function for calculating rejection function g
    inline CELER_FUNCTION real_type calc_g_fraction(real_type epsilon_min,
                                                    real_type epsilon_max);

    // Maximum energy fraction transferred to free electron [MeV]
    static CELER_CONSTEXPR_FUNCTION real_type max_energy_fraction()
    {
        return 1;
    }

};  // namespace BhabhaEnergyDistribution

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with data from MollerBhabhaInteractor.
 */
CELER_FUNCTION
BhabhaEnergyDistribution::BhabhaEnergyDistribution(Mass electron_mass,
                                                   Energy min_valid_energy,
                                                   Energy inc_energy)
    : min_energy_fraction_(value_as<Energy>(min_valid_energy)
                           / value_as<Energy>(inc_energy))
    , gamma_(1 + value_as<Energy>(inc_energy) / value_as<Mass>(electron_mass))
{
    CELER_EXPECT(electron_mass > zero_quantity()
                 && inc_energy > zero_quantity());
}

//---------------------------------------------------------------------------//
/*!
 * Sample epsilon for Bhabha scattering.
 */
template<class Engine>
CELER_FUNCTION real_type BhabhaEnergyDistribution::operator()(Engine& rng)
{
    real_type const g_denominator = this->calc_g_fraction(
        min_energy_fraction_, this->max_energy_fraction());

    UniformRealDistribution<> sample_inverse_epsilon(
        1 / this->max_energy_fraction(), 1 / min_energy_fraction_);

    // Sample epsilon
    real_type epsilon;
    do
    {
        epsilon = 1 / sample_inverse_epsilon(rng);
    } while (RejectionSampler<>(this->calc_g_fraction(epsilon, epsilon),
                                g_denominator)(rng));

    return epsilon;
}

//---------------------------------------------------------------------------//
/*!
 * Evaluate the rejection function g.
 */
CELER_FUNCTION real_type BhabhaEnergyDistribution::calc_g_fraction(
    real_type epsilon_min, real_type epsilon_max)
{
    real_type const y = 1 / (1 + gamma_);
    real_type const y_sq = ipow<2>(y);
    real_type const one_minus_2y = 1 - 2 * y;

    real_type const b1 = 2 - y_sq;
    real_type const b2 = one_minus_2y * (3 + y_sq);
    real_type const b4 = ipow<3>(one_minus_2y);
    real_type const b3 = ipow<2>(one_minus_2y) + b4;
    real_type const beta_sq = 1 - (1 / ipow<2>(gamma_));

    return 1
           + (ipow<4>(epsilon_max) * b4 - ipow<3>(epsilon_min) * b3
              + ipow<2>(epsilon_max) * b2 - epsilon_min * b1)
                 * beta_sq;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
