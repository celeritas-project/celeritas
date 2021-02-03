//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MollerEnergyDistribution.i.hh
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
MollerEnergyDistribution::MollerEnergyDistribution(
    const MollerBhabhaPointers& shared, const real_type inc_energy)
    : electron_mass_c_sq_(shared.electron_mass_c_sq)
    , inc_energy_(inc_energy)
    , total_energy_(inc_energy + shared.electron_mass_c_sq)
    , max_energy_fraction_(0.5)
    , min_energy_fraction_(shared.min_valid_energy / inc_energy)
    , gamma_(total_energy_ / shared.electron_mass_c_sq)
{
    CELER_EXPECT(electron_mass_c_sq_ > 0 && inc_energy_ > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Sample epsilon for Moller scattering.
 */
template<class Engine>
CELER_FUNCTION real_type MollerEnergyDistribution::operator()(Engine& rng)
{
    const real_type rejection_g = this->calc_f_g(max_energy_fraction_);
    UniformRealDistribution<> sample_inverse_epsilon(1 / max_energy_fraction_,
                                                     1 / min_energy_fraction_);

    // Sample epsilon
    real_type prob_f;
    real_type epsilon;
    do
    {
        epsilon = 1 / sample_inverse_epsilon(rng);
        prob_f  = calc_f_g(epsilon);

    } while (BernoulliDistribution(prob_f / rejection_g)(rng));

    return epsilon;
}

//---------------------------------------------------------------------------//
/*
 * Calculate probability density function f or rejection function g.
 */
CELER_FUNCTION real_type MollerEnergyDistribution::calc_f_g(real_type epsilon)
{
    const real_type two_gamma_term  = (2.0 * gamma_ - 1.0) / ipow<2>(gamma_);
    const real_type complement_frac = 1.0 - epsilon;

    return 1.0 - two_gamma_term * epsilon
           + ipow<2>(epsilon)
                 * (1.0 - two_gamma_term
                    + (1.0 - two_gamma_term * complement_frac)
                          / ipow<2>(complement_frac));
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
