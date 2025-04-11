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
#include "corecel/random/distribution/GenerateCanonical.hh"
#include "corecel/random/distribution/detail/GenerateCanonical32.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store several PRNG engine results and return them.
 */
template<class Engine, size_type Bytes>
class CachedRngEngine
{
    static_assert(Bytes > 0);

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
    static CELER_CONSTEXPR_FUNCTION size_type size() { return num_entries_; }

    //! Get the number of remaining samples
    CELER_CONSTEXPR_FUNCTION size_type remaining() const
    {
        return num_entries_ - next_;
    }

  private:
    static constexpr size_type num_entries_ = Bytes / sizeof(result_type);

    static_assert(num_entries_ * sizeof(result_type) == Bytes,
                  "number of bytes must be divisible by engine result size");

    /// DATA ///

    Array<result_type, num_entries_> stored_;
    size_type next_{0};
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Return an RNG with the given number of bytes cached.
 */
template<size_type Bytes, class Engine>
inline auto cache_rng_bytes(Engine& e)
{
    return CachedRngEngine<Engine, Bytes>{e};
}

//---------------------------------------------------------------------------//
/*!
 * Return an RNG with enough space to return Count of type T.
 */
template<class T, size_type Count, class Engine>
inline auto cache_rng_values(Engine& e)
{
    return CachedRngEngine<Engine, sizeof(T) * Count>{e};
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
