//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/distribution/GenerateCanonical.hh
//---------------------------------------------------------------------------//
#pragma once

#include <limits>
#include <random>
#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Helper function to generate a random uniform number
template<class RealType, class Generator>
inline CELER_FUNCTION RealType generate_canonical(Generator& g);

//---------------------------------------------------------------------------//
//! Sample a real_type on [0, 1).
template<class Generator>
inline CELER_FUNCTION real_type generate_canonical(Generator& g);

//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1).
 *
 * This is essentially an implementation detail; partial specialization can be
 * used to sample using special functions with a given generator.
 */
template<class Generator, class RealType = ::celeritas::real_type>
class GenerateCanonical
{
    static_assert(std::is_floating_point<RealType>::value,
                  "RealType must be float or double");

  public:
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

  public:
    // Sample a random number
    result_type operator()(Generator& rng);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1).
 *
 * This is the default implementation, for CPU-only code.
 */
template<class Generator, class RealType>
auto GenerateCanonical<Generator, RealType>::operator()(Generator& rng)
    -> result_type
{
    using limits_t = std::numeric_limits<result_type>;
    return std::generate_canonical<result_type, limits_t::digits>(rng);
}

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
}  // namespace celeritas
