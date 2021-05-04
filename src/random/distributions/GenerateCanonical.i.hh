//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GenerateCanonical.i.hh
//---------------------------------------------------------------------------//

#if !CELER_SHIELD_DEVICE
#    include <random>
#endif

namespace celeritas
{
#if !CELER_SHIELD_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1).
 *
 * This is the default implementation, for CPU only code.
 */
template<class Generator, class RealType>
auto GenerateCanonical<Generator, RealType>::operator()(Generator& rng)
    -> result_type
{
    using limits_t = std::numeric_limits<result_type>;
    return std::generate_canonical<result_type, limits_t::digits>(rng);
}
#endif

//---------------------------------------------------------------------------//
/*!
 * Helper function to generate a random real number in [0, 1).
 */
template<class RealType, class Generator>
CELER_FUNCTION RealType generate_canonical(Generator& g)
{
    return GenerateCanonical<Generator, RealType>()(g);
}

//---------------------------------------------------------------------------//
/*!
 * Helper function to generate a random real number in [0, 1).
 */
template<class Generator>
CELER_FUNCTION real_type generate_canonical(Generator& g)
{
    return GenerateCanonical<Generator, real_type>()(g);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
