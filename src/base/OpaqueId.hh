//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OpaqueId.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#ifndef __CUDA_ARCH__
#    include <functional>
#endif
#include "Assert.hh"
#include "Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Type-safe container for an integer identifier.
 *
 * \tparam Instantiator Class that uses the indexing type.
 * \tparam T Value type for the ID.
 *
 * This allows type-safe, read-only indexing/access for a class. The value is
 * 'true' if it's assigned, 'false' if invalid.
 */
template<class Instantiator, class T = unsigned int>
class OpaqueId
{
  public:
    //!@{
    //! Type aliases
    using instantiator_type = Instantiator;
    using value_type        = T;
    //!@}

  public:
    //! Default to invalid state
    CELER_CONSTEXPR_FUNCTION OpaqueId() : value_(OpaqueId::invalid_value()) {}

    //! Construct explicitly with stored value
    explicit CELER_CONSTEXPR_FUNCTION OpaqueId(value_type index)
        : value_(index)
    {
    }

    //! Whether this ID is in a valid (assigned) state
    explicit CELER_CONSTEXPR_FUNCTION operator bool() const
    {
        return value_ != invalid_value();
    }

    //! Get the ID's value
    CELER_FORCEINLINE_FUNCTION value_type get() const
    {
        CELER_EXPECT(*this);
        return value_;
    }

    //! Get the value without checking for validity (atypical)
    CELER_CONSTEXPR_FUNCTION value_type unchecked_get() const
    {
        return value_;
    }

  private:
    value_type value_;

    //// IMPLEMENTATION FUNCTIONS ////

    //! Value indicating the ID is not assigned
    static CELER_CONSTEXPR_FUNCTION value_type invalid_value()
    {
        return static_cast<value_type>(-1);
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Test equality
template<class I, class T>
CELER_CONSTEXPR_FUNCTION bool operator==(OpaqueId<I, T> lhs, OpaqueId<I, T> rhs)
{
    return lhs.unchecked_get() == rhs.unchecked_get();
}

//! Test inequality
template<class I, class T>
CELER_CONSTEXPR_FUNCTION bool operator!=(OpaqueId<I, T> lhs, OpaqueId<I, T> rhs)
{
    return !(lhs == rhs);
}

//! Allow less-than comparison for sorting
template<class I, class T>
CELER_CONSTEXPR_FUNCTION bool operator<(OpaqueId<I, T> lhs, OpaqueId<I, T> rhs)
{
    return lhs.unchecked_get() < rhs.unchecked_get();
}

//! Allow less-than comparison with *integer* for container comparison
template<class I, class T, class U>
CELER_CONSTEXPR_FUNCTION bool operator<(OpaqueId<I, T> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) < rhs);
}

//! Get the number of IDs enclosed by two opaque IDs.
template<class I, class T>
inline CELER_FUNCTION T operator-(OpaqueId<I, T> self, OpaqueId<I, T> other)
{
    CELER_EXPECT(self);
    CELER_EXPECT(other);
    return self.unchecked_get() - other.unchecked_get();
}

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
// STD::HASH SPECIALIZATION FOR HOST CODE
//---------------------------------------------------------------------------//
//! \cond
#ifndef __CUDA_ARCH__
namespace std
{
//! Specialization for std::hash for unordered storage.
template<class I, class T>
struct hash<celeritas::OpaqueId<I, T>>
{
    using argument_type = celeritas::OpaqueId<I, T>;
    using result_type   = std::size_t;
    result_type operator()(const argument_type& id) const noexcept
    {
        return std::hash<T>()(id.unchecked_get());
    }
};
} // namespace std
#endif // __CUDA_ARCH__
//! \endcond
