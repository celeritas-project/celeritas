//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TsaiUrbanDistribution.i.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Algorithms.hh"
#include "random/distributions/BernoulliDistribution.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from input data.
 */
CELER_FUNCTION
TsaiUrbanDistribution::TsaiUrbanDistribution(MevEnergy energy, MevMass mass)
{
    // MevMass{}.value() [MeV]
    umax_ = 2 * (1 + energy.value() / mass.value());
}

//---------------------------------------------------------------------------//
/*!
 * Sample gamma angle (z-axis along the parent particle).
 */
template<class Engine>
CELER_FUNCTION real_type TsaiUrbanDistribution::operator()(Engine& rng)
{
    real_type u;
    do
    {
        real_type uu
            = -std::log(generate_canonical(rng) * generate_canonical(rng));
        u = uu
            * (BernoulliDistribution(0.25)(rng) ? real_type(1.6)
                                                : real_type(1.6 / 3));
    } while (u > umax_);

    return 1 - 2 * ipow<2>(u / umax_);
}
//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
