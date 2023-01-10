//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/distribution/BhabhaEnergyDistribution.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for \c MollerBhabhaInteractor .
 *
 * Sample the exiting energy for Bhabha scattering.
 */
class BhabhaEnergyDistribution
{
  public:
    // Construct with data from MollerBhabhaInteractor
    inline CELER_FUNCTION BhabhaEnergyDistribution(real_type electron_mass_c_sq,
                                                   real_type min_valid_energy,
                                                   real_type inc_energy);

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
BhabhaEnergyDistribution::BhabhaEnergyDistribution(real_type electron_mass_c_sq,
                                                   real_type min_valid_energy,
                                                   real_type inc_energy)
    : inc_energy_(inc_energy)
    , total_energy_(inc_energy + electron_mass_c_sq)
    , min_energy_fraction_(min_valid_energy / inc_energy)
    , gamma_(total_energy_ / electron_mass_c_sq)
{
    CELER_EXPECT(electron_mass_c_sq > 0 && inc_energy_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample epsilon for Bhabha scattering.
 */
template<class Engine>
CELER_FUNCTION real_type BhabhaEnergyDistribution::operator()(Engine& rng)
{
    const real_type g_denominator = this->calc_g_fraction(
        min_energy_fraction_, this->max_energy_fraction());

    UniformRealDistribution<> sample_inverse_epsilon(
        1 / this->max_energy_fraction(), 1 / min_energy_fraction_);

    // Sample epsilon
    real_type g_numerator;
    real_type epsilon;
    do
    {
        epsilon = 1 / sample_inverse_epsilon(rng);
        g_numerator = this->calc_g_fraction(epsilon, epsilon);

    } while (BernoulliDistribution(g_numerator / g_denominator)(rng));

    return epsilon;
}

//---------------------------------------------------------------------------//
/*
 * Helper function for calculating rejection function g.
 */
CELER_FUNCTION real_type BhabhaEnergyDistribution::calc_g_fraction(
    real_type epsilon_min, real_type epsilon_max)
{
    const real_type y = 1.0 / (1.0 + gamma_);
    const real_type y_sq = ipow<2>(y);
    const real_type one_minus_2y = 1.0 - 2.0 * y;

    const real_type b1 = 2.0 - y_sq;
    const real_type b2 = one_minus_2y * (3.0 + y_sq);
    const real_type b4 = ipow<3>(one_minus_2y);
    const real_type b3 = ipow<2>(one_minus_2y) + b4;
    const real_type beta_sq = 1.0 - (1.0 / ipow<2>(gamma_));

    return 1.0
           + (ipow<4>(epsilon_max) * b4 - ipow<3>(epsilon_min) * b3
              + ipow<2>(epsilon_max) * b2 - epsilon_min * b1)
                 * beta_sq;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
