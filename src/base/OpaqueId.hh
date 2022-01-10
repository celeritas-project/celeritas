//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OpaqueId.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include "Assert.hh"
#include "Macros.hh"
#include "Types.hh"

namespace celeritas
{
// Forward declare meaningless struct to avoid conflict with globally
// namespaced PTL::Thread class when defining ThreadId type.
struct Thread;

//---------------------------------------------------------------------------//
/*!
 * Type-safe index for accessing an array.
 *
 * \tparam ValueT Type of each item in the array.
 * \tparam SizeT Integer index
 *
 * This allows type-safe, read-only indexing/access for a class. The value is
 * 'true' if it's assigned, 'false' if invalid.
 */
template<class ValueT, class SizeT = ::celeritas::size_type>
class OpaqueId
{
    static_assert(static_cast<SizeT>(-1) > 0, "SizeT must be unsigned.");

  public:
    //!@{
    //! Type aliases
    using value_type = ValueT;
    using size_type  = SizeT;
    //!@}

  public:
    //! Default to invalid state
    CELER_CONSTEXPR_FUNCTION OpaqueId() : value_(OpaqueId::invalid_value()) {}

    //! Construct explicitly with stored value
    explicit CELER_CONSTEXPR_FUNCTION OpaqueId(size_type index) : value_(index)
    {
    }

    //! Whether this ID is in a valid (assigned) state
    explicit CELER_CONSTEXPR_FUNCTION operator bool() const
    {
        return value_ != invalid_value();
    }

    //! Get the ID's value
    CELER_FORCEINLINE_FUNCTION size_type get() const
    {
        CELER_EXPECT(*this);
        return value_;
    }

    //! Get the value without checking for validity (atypical)
    CELER_CONSTEXPR_FUNCTION size_type unchecked_get() const { return value_; }

  private:
    size_type value_;

    //// IMPLEMENTATION FUNCTIONS ////

    //! Value indicating the ID is not assigned
    static CELER_CONSTEXPR_FUNCTION size_type invalid_value()
    {
        return static_cast<size_type>(-1);
    }
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Test equality
template<class V, class S>
CELER_CONSTEXPR_FUNCTION bool operator==(OpaqueId<V, S> lhs, OpaqueId<V, S> rhs)
{
    return lhs.unchecked_get() == rhs.unchecked_get();
}

//! Test inequality
template<class V, class S>
CELER_CONSTEXPR_FUNCTION bool operator!=(OpaqueId<V, S> lhs, OpaqueId<V, S> rhs)
{
    return !(lhs == rhs);
}

//! Allow less-than comparison for sorting
template<class V, class S>
CELER_CONSTEXPR_FUNCTION bool operator<(OpaqueId<V, S> lhs, OpaqueId<V, S> rhs)
{
    return lhs.unchecked_get() < rhs.unchecked_get();
}

//! Allow less-than comparison with *integer* for container comparison
template<class V, class S, class U>
CELER_CONSTEXPR_FUNCTION bool operator<(OpaqueId<V, S> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) < rhs);
}

//! Get the number of IDs enclosed by two opaque IDs.
template<class V, class S>
inline CELER_FUNCTION S operator-(OpaqueId<V, S> self, OpaqueId<V, S> other)
{
    CELER_EXPECT(self);
    CELER_EXPECT(other);
    return self.unchecked_get() - other.unchecked_get();
}

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Index of the current CUDA thread, using an imaginary class named Thread.
using ThreadId = OpaqueId<struct Thread>;

//---------------------------------------------------------------------------//
} // namespace celeritas

//---------------------------------------------------------------------------//
// STD::HASH SPECIALIZATION FOR HOST CODE
//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//! Specialization for std::hash for unordered storage.
template<class V, class S>
struct hash<celeritas::OpaqueId<V, S>>
{
    using argument_type = celeritas::OpaqueId<V, S>;
    using result_type   = std::size_t;
    result_type operator()(const argument_type& id) const noexcept
    {
        return std::hash<S>()(id.unchecked_get());
    }
};
} // namespace std
//! \endcond
