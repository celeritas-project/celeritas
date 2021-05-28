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
MollerEnergyDistribution::MollerEnergyDistribution(real_type electron_mass_c_sq,
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
 * Sample epsilon for Moller scattering.
 */
template<class Engine>
CELER_FUNCTION real_type MollerEnergyDistribution::operator()(Engine& rng)
{
    const real_type g_denominator
        = this->calc_g_fraction(this->max_energy_fraction());

    UniformRealDistribution<> sample_inverse_epsilon(
        1 / this->max_energy_fraction(), 1 / min_energy_fraction_);

    // Sample epsilon
    real_type g_numerator;
    real_type epsilon;
    do
    {
        epsilon     = 1 / sample_inverse_epsilon(rng);
        g_numerator = calc_g_fraction(epsilon);

    } while (BernoulliDistribution(g_numerator / g_denominator)(rng));

    return epsilon;
}

//---------------------------------------------------------------------------//
/*
 * Helper function for calculating rejection function g.
 */
CELER_FUNCTION real_type
MollerEnergyDistribution::calc_g_fraction(real_type epsilon)
{
    const real_type two_gamma_term  = (2 * gamma_ - 1) / ipow<2>(gamma_);
    const real_type complement_frac = 1 - epsilon;

    return 1 - two_gamma_term * epsilon
           + ipow<2>(epsilon)
                 * (1 - two_gamma_term
                    + (1 - two_gamma_term * complement_frac)
                          / ipow<2>(complement_frac));
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
