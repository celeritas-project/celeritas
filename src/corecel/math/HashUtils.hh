//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/HashUtils.hh
// TODO for v1.0: rename to Hasher.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <type_traits>

#include "corecel/Macros.hh"
#include "corecel/cont/Span.hh"

#include "detail/FnvHasher.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TODO: add CMake configuration argument so this can be swapped out with e.g.
// xxHash
using Hasher = detail::FnvHasher<std::size_t>;

//---------------------------------------------------------------------------//
/*!
 * Hash a span of contiguous data without padding.
 *
 * This should generally only be used if \c has_unique_object_representations_v
 * is \c true, because e.g. structs have padding so this may result in reading
 * uninitialized data or giving two equal structs different hashes.
 */
template<class T, std::size_t N>
std::size_t hash_as_bytes(Span<T const, N> s)
{
    std::size_t result{};
    Hasher hash{&result};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    hash(Span<std::byte const>{reinterpret_cast<std::byte const*>(s.data()),
                               s.size() * sizeof(T)});
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Combine hashes of the given arguments using a fast hash algorithm.
 *
 * See https://florianjw.de/en/variadic_templates.html for why we constructed
 * this as such. By making the variadic template use function argument
 * expansion rather than recursion, we can unpack the args in a left-to-right
 * order. The `(HASH,0)` construction is to give the unpacked expression a
 * return type; and putting these in an `initializer_list` constructor
 * guarantees the hashes are evaluated from left to right (unlike a typical
 * argument expansion where the orders may be arbitrary).
 */
template<class... Args>
std::size_t hash_combine(Args const&... args)
{
    // Construct a hasher and initialize
    std::size_t result{};
    [[maybe_unused]] Hasher hash{&result};

    // Hash each one of the arguments sequentially by expanding into an unused
    // initializer list.
    (void)std::initializer_list<int>{(hash(std::hash<Args>()(args)), 0)...};

    return result;
}
//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//---------------------------------------------------------------------------//
/*!
 * Hash specialization for celeritas span.
 *
 * This has three distinct cases:
 * 1. Unique object representation: use for structs without padding or floats.
 * 2. Floating point values: these will hash uniquely if they're not NaN.
 * 3. All other types: combine the hash of their individual values.
 */
template<class T, std::size_t Extent>
struct hash<celeritas::Span<T, Extent>>
{
    std::size_t operator()(celeritas::Span<T, Extent> const& s) const
    {
        if constexpr (std::has_unique_object_representations_v<T>)
        {
            return celeritas::hash_as_bytes(s);
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            CELER_EXPECT(([&s] {
                for (auto const& v : s)
                {
                    if (v != v)
                        return false;
                }
                return true;
            }()));
            return celeritas::hash_as_bytes(s);
        }
        else
        {
            std::size_t result{};
            celeritas::Hasher hash_impl{&result};
            for (auto const& v : s)
            {
                hash_impl(std::hash<std::remove_cv_t<T>>{}(v));
            }
            return result;
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace std
//! \endcond
