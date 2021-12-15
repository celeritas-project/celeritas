//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SBEnergyDistribution.i.hh
//---------------------------------------------------------------------------//
#include <cmath>

#include "base/Algorithms.hh"
#include "random/distributions/BernoulliDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from incident particle and energy.
 *
 * The incident energy *must* be within the bounds of the SB table data, so the
 * Model's applicability must be consistent with the table data.
 */
template<class X>
CELER_FUNCTION
SBEnergyDistribution<X>::SBEnergyDistribution(const SBEnergyDistHelper& helper,
                                              X scale_xs)
    : helper_(helper)
    , inv_max_xs_{1
                  / (helper.max_xs().value() * scale_xs(helper.max_xs_energy()))}
    , scale_xs_(::celeritas::move(scale_xs))
{
}

//---------------------------------------------------------------------------//
/*!
 * Sample the exiting energy by doing a table lookup and rejection.
 */
template<class X>
template<class Engine>
CELER_FUNCTION auto SBEnergyDistribution<X>::operator()(Engine& rng) -> Energy
{
    // Sampled energy
    Energy exit_energy;
    // Calculated cross section used inside rejection sampling
    real_type xs{};
    do
    {
        // Sample scaled energy and subtract correction factor
        exit_energy = helper_.sample_exit_energy(rng);

        // Interpolate the differential cross setion at the sampled exit energy
        xs = helper_.calc_xs(exit_energy).value() * scale_xs_(exit_energy);
        CELER_ASSERT(xs >= 0 && xs <= 1 / inv_max_xs_);
    } while (!BernoulliDistribution(xs * inv_max_xs_)(rng));
    return exit_energy;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
