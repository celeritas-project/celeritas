//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GenerateCanonical32.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1) from a 32-bit generator.
 */
template<class RealType = ::celeritas::real_type>
class GenerateCanonical32;

template<>
class GenerateCanonical32<float>
{
  public:
    //!@{
    //! Type aliases
    using result_type = float;
    //!@}

  public:
    //! Sample a random number with floating point precision
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);
};

template<>
class GenerateCanonical32<double>
{
  public:
    //!@{
    //! Type aliases
    using result_type = double;
    //!@}

  public:
    //! Sample a random number with floating point precision
    template<class Generator>
    inline CELER_FUNCTION result_type operator()(Generator& rng);
};

//---------------------------------------------------------------------------//
/*!
 * Generate a 32-bit float from one 32-bit sample.
 *
 * Returns a float on [0, 1) assuming a generator range of [0, 2^32).
 */
template<class Generator>
CELER_FUNCTION float GenerateCanonical32<float>::operator()(Generator& rng)
{
    using uint_t = typename Generator::result_type;
    static_assert(sizeof(uint_t) == 4, "Generator value must be 32-bit");
    static_assert(sizeof(typename Generator::result_type) == 4,
                  "Generator must return 32-bit value");

    constexpr float norm = 2.32830643654e-10f; // 1 / 2**32
    return norm * rng();
}

//---------------------------------------------------------------------------//
/*!
 * Generate a 64-bit double from two 32-bit samples.
 *
 * Returns a double on [0, 1).
 */
template<class Generator>
CELER_FUNCTION double GenerateCanonical32<double>::operator()(Generator& rng)
{
    using uint_t = typename Generator::result_type;
    static_assert(sizeof(uint_t) == 4, "Generator value must be 32-bit");
    static_assert(sizeof(ull_int) == 8, "Expected 64-bit UL");

    unsigned int upper = rng();
    unsigned int lower = rng();

    // Convert the two 32-byte samples to a 53-bit-precision double by shifting
    // the 'upper' and combining the lower bits of the upper with the higher
    // bits of the lower
    constexpr double norm = 1.1102230246251565e-16; // 1 / 2^53
    return norm
           * ((static_cast<ull_int>(upper) << (53ul - 32ul))
              ^ static_cast<ull_int>(lower));
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
