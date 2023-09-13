//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/OpaqueId.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>

#include "Assert.hh"
#include "Macros.hh"
#include "Types.hh"

namespace celeritas
{
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
    //! \name Type aliases
    using value_type = ValueT;
    using size_type = SizeT;
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

    //! Pre-increment of the ID
    CELER_FUNCTION OpaqueId& operator++()
    {
        CELER_EXPECT(*this);
        value_ += 1;
        return *this;
    }

    //! Post-increment of the ID
    CELER_FUNCTION OpaqueId operator++(int)
    {
        CELER_EXPECT(*this);
        OpaqueId old{*this};
        ++*this;
        return old;
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
#define CELER_DEFINE_OPAQUEID_CMP(TOKEN)                             \
    template<class V, class S>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(OpaqueId<V, S> lhs, \
                                                 OpaqueId<V, S> rhs) \
    {                                                                \
        return lhs.unchecked_get() TOKEN rhs.unchecked_get();        \
    }

//!@{
//! Comparison for OpaqueId
CELER_DEFINE_OPAQUEID_CMP(==)
CELER_DEFINE_OPAQUEID_CMP(!=)
CELER_DEFINE_OPAQUEID_CMP(<)
CELER_DEFINE_OPAQUEID_CMP(>)
CELER_DEFINE_OPAQUEID_CMP(<=)
CELER_DEFINE_OPAQUEID_CMP(>=)
//!@}

#undef CELER_DEFINE_OPAQUEID_CMP

//! Allow less-than comparison with *integer* for container comparison
template<class V, class S, class U>
CELER_CONSTEXPR_FUNCTION bool operator<(OpaqueId<V, S> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) < rhs);
}

//! Allow less-than-equal comparison with *integer* for container comparison
template<class V, class S, class U>
CELER_CONSTEXPR_FUNCTION bool operator<=(OpaqueId<V, S> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) <= rhs);
}

//! Get the distance between two opaque IDs
template<class V, class S>
inline CELER_FUNCTION S operator-(OpaqueId<V, S> self, OpaqueId<V, S> other)
{
    CELER_EXPECT(self);
    CELER_EXPECT(other);
    return self.unchecked_get() - other.unchecked_get();
}

//! Increment an opaque ID by an offset
template<class V, class S>
inline CELER_FUNCTION OpaqueId<V, S>
operator+(OpaqueId<V, S> id, std::make_signed_t<S> offset)
{
    CELER_EXPECT(id);
    CELER_EXPECT(offset >= 0 || static_cast<S>(-offset) <= id.unchecked_get());
    return OpaqueId<V, S>{id.unchecked_get() + static_cast<S>(offset)};
}

//! Decrement an opaque ID by an offset
template<class V, class S>
inline CELER_FUNCTION OpaqueId<V, S>
operator-(OpaqueId<V, S> id, std::make_signed_t<S> offset)
{
    CELER_EXPECT(id);
    CELER_EXPECT(offset <= 0 || static_cast<S>(offset) <= id.unchecked_get());
    return OpaqueId<V, S>{id.unchecked_get() - static_cast<S>(offset)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//! Specialization for std::hash for unordered storage.
template<class V, class S>
struct hash<celeritas::OpaqueId<V, S>>
{
    using argument_type = celeritas::OpaqueId<V, S>;
    using result_type = std::size_t;
    result_type operator()(argument_type const& id) const noexcept
    {
        return std::hash<S>()(id.unchecked_get());
    }
};
}  // namespace std
//! \endcond
