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
//! Implementation of each GenerateCanonical, used for CachedRngEngine
enum class GenerateCanonicalPolicy
{
    std,  //!< Use standard library
    builtin32,  //!< Use detail::GenerateCanonical32
    builtin64,  //!< Use detail::GenerateCanonical64 (not yet implemented)
    custom,  //!< Custom method
    size_
};

//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1).
 *
 * This is essentially an implementation detail; partial specialization can be
 * used to sample using special functions with a given generator.
 *
 * \todo For 1.0 refactor so that if RNGs have a builtin policy we call our
 * builtin functions; if they have a custom we call a function on the RNG
 * itself; and if they don't define a policy (e.g. a standard library
 * one) we fall back to calling std::generate_canonical.
 */
template<class Generator, class RealType = ::celeritas::real_type>
struct GenerateCanonical
{
    static_assert(std::is_floating_point<RealType>::value,
                  "RealType must be float or double");

    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = real_type;
    //!@}

    //! By default use standard library
    static constexpr auto policy = GenerateCanonicalPolicy::std;

    // Sample a random number
    inline result_type operator()(Generator& rng);
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
