//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/engine/CachedRngEngine.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/NumericLimits.hh"
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/detail/GenerateCanonical32.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store several PRNG engine results and return them.
 */
template<class Engine, size_type N>
class CachedRngEngine
{
    static_assert(N > 0);

  public:
    //!@{
    //! \name Type aliases
    using result_type = typename Engine::result_type;
    //!@}

  public:
    //! Lowest value potentially generated
    static CELER_CONSTEXPR_FUNCTION result_type min() { return Engine::min(); }
    //! Highest value potentially generated
    static CELER_CONSTEXPR_FUNCTION result_type max() { return Engine::max(); }

    // Save values on construction
    inline CELER_FUNCTION CachedRngEngine(Engine& e);

    // Return the next pseudorandom number in the sequence
    inline CELER_FUNCTION result_type operator()();

    //! Get the total number of stored samples
    static CELER_CONSTEXPR_FUNCTION size_type size() { return N; }

    //! Get the number of remaining samples
    CELER_CONSTEXPR_FUNCTION size_type remaining() const { return N - next_; }

  private:
    /// DATA ///

    Array<result_type, N> stored_;
    size_type next_{0};
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Return an RNG with the given number of calls cached.
 */
template<size_type N, class Engine>
inline CELER_FUNCTION auto cache_rng_count(Engine& e)
{
    return CachedRngEngine<Engine, N>{e};
}

//---------------------------------------------------------------------------//
/*!
 * Return an RNG with enough space to return Count of type T.
 */
template<class T, size_type Count, class Engine>
inline CELER_FUNCTION auto cache_rng_values(Engine& e)
{
    // Account for the fact that some implementations use 64-bit integers for
    // RNGs that have return 32 bits of entropy
    using result_type = typename Engine::result_type;
    static_assert(sizeof(result_type) == 4 || sizeof(result_type) == 8,
                  "only implemented for 32- and 64-bit integers");
    constexpr size_type bytes_entropy
        = sizeof(result_type) == 4                           ? 4
          : Engine::max() <= numeric_limits<unsigned>::max() ? sizeof(unsigned)
          : Engine::max() <= numeric_limits<unsigned long long>::max()
              ? sizeof(unsigned long long)
              : 0;
    return CachedRngEngine<Engine, Count * sizeof(T) / bytes_entropy>{e};
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Save values on construction.
 */
template<class Engine, size_type Bytes>
CELER_FUNCTION CachedRngEngine<Engine, Bytes>::CachedRngEngine(Engine& rng)
{
    for (result_type& entry : stored_)
    {
        entry = rng();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return the next pseudorandom number in the sequence.
 */
template<class Engine, size_type Bytes>
CELER_FUNCTION auto CachedRngEngine<Engine, Bytes>::operator()() -> result_type
{
    CELER_EXPECT(this->remaining() != 0);
    return stored_[next_++];
}

//---------------------------------------------------------------------------//
// SPECIALIZATIONS
//---------------------------------------------------------------------------//
/*!
 * Specialization of GenerateCanonical for cached engine.
 */
template<class Engine, size_type Bytes, class RealType>
struct GenerateCanonical<CachedRngEngine<Engine, Bytes>, RealType>
{
    //!@{
    //! \name Type aliases
    using real_type = RealType;
    using result_type = RealType;
    using argument_type = CachedRngEngine<Engine, Bytes>;
    //!@}

    // Decide what policy to use
    static constexpr auto policy = GenerateCanonical<Engine>::policy;

    //! Sample a random number on [0, 1)
    CELER_FORCEINLINE_FUNCTION result_type operator()(argument_type& rng)
    {
        if constexpr (policy == GenerateCanonicalPolicy::builtin32)
        {
            return detail::GenerateCanonical32<RealType>()(rng);
        }
        else if constexpr (policy == GenerateCanonicalPolicy::std)
        {
#ifndef CELER_DEVICE_SOURCE
            using limits_t = std::numeric_limits<result_type>;
            return std::generate_canonical<result_type, limits_t::digits>(rng);
#else
            CELER_ASSERT_UNREACHABLE();
#endif
        }
        else
        {
            CELER_NOT_IMPLEMENTED("Custom sampling for cached RNG engine");
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
