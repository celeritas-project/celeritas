//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/FnvHasher.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Implementation of the FNV-1a algorithm.
 *
 * From http://www.isthe.com/chongo/tech/comp/fnv:
 * <blockquote>
   The basis of the FNV hash algorithm was taken from an idea sent as
   reviewer comments to the IEEE POSIX P1003.2 committee by Glenn Fowler and
   Phong Vo back in 1991. In a subsequent ballot round: Landon Curt Noll
   improved on their algorithm. Some people tried this hash and found that it
   worked rather well. In an EMail message to Landon, they named it the
   "Fowler/Noll/Vo" or FNV hash.

   FNV hashes are designed to be fast while maintaining a low collision rate.
   The FNV speed allows one to quickly hash lots of data while maintaining a
   reasonable collision rate. The high dispersion of the FNV hashes makes
   them well suited for hashing nearly identical strings such as URLs,
   hostnames, filenames, text, IP addresses, etc.
   </blockquote>
 *
 * High dispersion, quick hashes for similar data is often needed for hash
 * tables of pairs of similar values (or strings).
 *
 * This hashing algorithm is *not* very fast: each byte requires an integer
 * addition and multiply!
 */
template<std::size_t S>
struct FnvHashTraits;

// 32-bit specialization
template<>
struct FnvHashTraits<4ul>
{
    using value_type = std::uint32_t;
    static constexpr value_type initial_basis = 0x811c9dc5u;
    static constexpr value_type magic_prime = 0x01000193u;
};

// 64-bit specialization
template<>
struct FnvHashTraits<8ul>
{
    using value_type = std::uint64_t;
    static constexpr value_type initial_basis = 0xcbf29ce484222325ull;
    static constexpr value_type magic_prime = 0x00000100000001b3ull;
};

//---------------------------------------------------------------------------//
/*!
 * Use a fast algorithm to construct a well-distributed hash.
 *
 * \tparam T integer type to use for hashing.
 *
 * This utility class is meant for processing keys in hash tables with native
 * integer size.
 */
template<class T>
class FnvHasher
{
    static_assert(std::is_unsigned<T>::value,
                  "Hash type must be an unsigned integer");

  public:
    //@{
    //! \name Type aliases
    using value_type = T;
    //@}

  public:
    // Construct with a reference to the hashed value which we initialize
    explicit inline FnvHasher(value_type* hash_result);

    // Hash a byte of data
    CELER_FORCEINLINE void operator()(std::byte b) const;

    // Hash a size_t (useful for std::hash integration)
    inline void operator()(std::size_t value) const;

    // Hash a span of bytes
    inline void operator()(Span<std::byte const> s) const;

  private:
    using TraitsT = FnvHashTraits<sizeof(T)>;

    // Current hash, starting with a prescribed initial value
    value_type* hash_;
};

//---------------------------------------------------------------------------//
/*!
 * Initialize the result on construction.
 */
template<class T>
FnvHasher<T>::FnvHasher(value_type* hash_result) : hash_(hash_result)
{
    CELER_EXPECT(hash_);
    *hash_ = TraitsT::initial_basis;
}

//---------------------------------------------------------------------------//
/*!
 * Hash a byte of data.
 *
 * The FNV1a algorithm is very simple.
 */
template<class T>
CELER_FORCEINLINE void FnvHasher<T>::operator()(std::byte b) const
{
    // XOR hash with the current byte
    *hash_ ^= std::to_integer<T>(b);
    // Multiply by magic prime
    *hash_ *= TraitsT::magic_prime;
}

//---------------------------------------------------------------------------//
/*!
 * Hash a size_t.
 *
 * This is useful for std::hash integration.
 */
template<class T>
void FnvHasher<T>::operator()(std::size_t value) const
{
    for (std::size_t i = 0; i < sizeof(std::size_t); ++i)
    {
        (*this)(static_cast<std::byte>(value & 0xffu));
        value >>= 8;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Hash a span of bytes.
 */
template<class T>
void FnvHasher<T>::operator()(Span<std::byte const> bytes) const
{
    for (auto b : bytes)
    {
        (*this)(b);
    }
}

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class T>
FnvHasher(T*) -> FnvHasher<T>;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
