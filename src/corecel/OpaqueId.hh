//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Type-safe index for accessing an array or collection of data.
 *
 * \tparam ValueT Type of each item in an array
 * \tparam SizeT Unsigned integer index
 *
 * It's common for classes and functions to take multiple indices, especially
 * for O(1) indexing for performance. By annotating these values with a type,
 * we give them semantic meaning, and we gain compile-time type safety.
 *
 * If this class is used for indexing into an array, then \c ValueT argument
 * should usually be the value type of the array:
 * <code>Foo operator[](OpaqueId<Foo>)</code>
 *
 * An \c OpaqueId object evaluates to \c true if it has a value, or \c false if
 * it does not (i.e., it has an "invalid" value).
 *
 * See also \c id_cast below for checked construction of OpaqueIds from generic
 * integer values (avoid compile-time warnings or errors from signed/truncated
 * integers).
 */
template<class ValueT, class SizeT = ::celeritas::size_type>
class OpaqueId
{
    static_assert(std::is_unsigned_v<SizeT> && !std::is_same_v<SizeT, bool>,
                  "SizeT must be unsigned.");

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
        OpaqueId old{*this};
        ++*this;
        return old;
    }

    //! Pre-decrement of the ID
    CELER_FUNCTION OpaqueId& operator--()
    {
        CELER_EXPECT(*this && value_ > 0);
        value_ -= 1;
        return *this;
    }

    //! Post-decrement of the ID
    CELER_FUNCTION OpaqueId operator--(int)
    {
        OpaqueId old{*this};
        --*this;
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

    //! Access the underlying data for more efficient loading from memory
    CELER_CONSTEXPR_FUNCTION size_type const* data() const { return &value_; }

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
/*!
 * Safely create an OpaqueId from an integer of any type.
 *
 * This asserts that the integer is in the \em valid range of the target ID
 * type, and casts to it.
 *
 * \note The value cannot be the underlying "invalid" value, i.e.
 * <code> static_cast<FooId>(FooId{}.unchecked_get()) </code> will not work.
 */
template<class IdT, class T>
inline CELER_FUNCTION IdT id_cast(T value) noexcept(!CELERITAS_DEBUG)
{
    static_assert(std::is_integral_v<T>);
    if constexpr (!std::is_unsigned_v<T>)
    {
        CELER_EXPECT(value >= 0);
    }

    using IdSize = typename IdT::size_type;
    if constexpr (!std::is_same_v<T, IdSize>)
    {
        // Check that value is within the integer range [0, N-1)
        using U = std::common_type_t<IdSize, std::make_unsigned_t<T>>;
        CELER_EXPECT(static_cast<U>(value)
                     < static_cast<U>(IdT{}.unchecked_get()));
    }
    else
    {
        // Check that value is *not* the invalid value
        CELER_EXPECT(value != IdT{}.unchecked_get());
    }

    return IdT{static_cast<IdSize>(value)};
}

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

//---------------------------------------------------------------------------//
//! Allow less-than comparison with *integer* for container comparison
template<class V, class S, class U>
CELER_CONSTEXPR_FUNCTION bool operator<(OpaqueId<V, S> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) < rhs);
}

//---------------------------------------------------------------------------//
//! Allow less-than-equal comparison with *integer* for container comparison
template<class V, class S, class U>
CELER_CONSTEXPR_FUNCTION bool operator<=(OpaqueId<V, S> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) <= rhs);
}

//---------------------------------------------------------------------------//
//! Get the distance between two opaque IDs
template<class V, class S>
inline CELER_FUNCTION S operator-(OpaqueId<V, S> self, OpaqueId<V, S> other)
{
    CELER_EXPECT(self);
    CELER_EXPECT(other);
    return self.unchecked_get() - other.unchecked_get();
}

//---------------------------------------------------------------------------//
//! Increment an opaque ID by an offset
template<class V, class S>
inline CELER_FUNCTION OpaqueId<V, S>
operator+(OpaqueId<V, S> id, std::make_signed_t<S> offset)
{
    CELER_EXPECT(id);
    CELER_EXPECT(offset >= 0 || static_cast<S>(-offset) <= id.unchecked_get());
    return OpaqueId<V, S>{id.unchecked_get() + static_cast<S>(offset)};
}

//---------------------------------------------------------------------------//
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
    std::size_t operator()(celeritas::OpaqueId<V, S> const& id) const noexcept
    {
        return std::hash<S>()(id.unchecked_get());
    }
};
}  // namespace std
//! \endcond
