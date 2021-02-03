//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file BhabhaEnergyDistribution.i.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "random/distributions/UniformRealDistribution.hh"
#include "random/distributions/BernoulliDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with data from MollerBhabhaInteractor.
 */
CELER_FUNCTION
BhabhaEnergyDistribution::BhabhaEnergyDistribution(
    const MollerBhabhaPointers& shared, const real_type inc_energy)
    : electron_mass_c_sq_(shared.electron_mass_c_sq)
    , inc_energy_(inc_energy)
    , total_energy_(inc_energy + shared.electron_mass_c_sq)
    , max_energy_fraction_(1.0)
    , min_energy_fraction_(shared.min_valid_energy / inc_energy)
    , gamma_(total_energy_ / shared.electron_mass_c_sq)
{
    CELER_EXPECT(electron_mass_c_sq_ > 0 && inc_energy_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample epsilon for Bhabha scattering.
 */
template<class Engine>
CELER_FUNCTION real_type BhabhaEnergyDistribution::operator()(Engine& rng)
{
    const real_type rejection_g
        = this->calc_f_g(min_energy_fraction_, max_energy_fraction_);

    UniformRealDistribution<> sample_inverse_epsilon(1 / max_energy_fraction_,
                                                     1 / min_energy_fraction_);

    // Sample epsilon
    real_type prob_f;
    real_type epsilon;
    do
    {
        epsilon = 1 / sample_inverse_epsilon(rng);
        prob_f  = this->calc_f_g(epsilon, epsilon);

    } while (BernoulliDistribution(prob_f / rejection_g)(rng));

    return epsilon;
}

//---------------------------------------------------------------------------//
/*
 * Calculate probability density function f or rejection function g.
 */
CELER_FUNCTION real_type BhabhaEnergyDistribution::calc_f_g(
    real_type epsilon_min, real_type epsilon_max)
{
    const real_type y            = 1.0 / (1.0 + gamma_);
    const real_type y_sq         = ipow<2>(y);
    const real_type one_minus_2y = 1.0 - 2.0 * y;

    const real_type b1      = 2.0 - y_sq;
    const real_type b2      = one_minus_2y * (3.0 + y_sq);
    const real_type b4      = ipow<3>(one_minus_2y);
    const real_type b3      = ipow<2>(one_minus_2y) + b4;
    const real_type beta_sq = 1.0 - (1.0 / ipow<2>(gamma_));

    return 1.0
           + (ipow<4>(epsilon_max) * b4 - ipow<3>(epsilon_min) * b3
              + ipow<2>(epsilon_max) * b2 - epsilon_min * b1)
                 * beta_sq;
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
