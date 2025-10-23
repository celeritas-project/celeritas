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
namespace detail
{
//---------------------------------------------------------------------------//
//! Sentinel value for an unassigned opaque ID
template<class T>
inline constexpr T nullid_value{static_cast<T>(-1)};

//---------------------------------------------------------------------------//
//! Safely cast from one integer T to another U, avoiding the sentinel value
template<class T, class U>
inline CELER_FUNCTION T id_cast_impl(U value) noexcept(!CELERITAS_DEBUG)
{
    if constexpr (std::is_signed_v<U>)
    {
        CELER_EXPECT(value >= 0);
    }

    if constexpr (!std::is_same_v<T, U>)
    {
        // Check that the cast value is within the integer range [0, N-1)
        using C = std::common_type_t<T, std::make_unsigned_t<U>>;
        if constexpr (std::is_signed_v<C>)
        {
            CELER_EXPECT(static_cast<C>(value) >= 0);
        }
        CELER_EXPECT(static_cast<C>(value) < static_cast<C>(nullid_value<T>));
    }
    else
    {
        // Check that value is *not* the null value
        CELER_EXPECT(static_cast<T>(value) != nullid_value<T>);
    }

    return static_cast<T>(value);
}

//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Type-safe index for accessing an array or collection of data.
 *
 * \tparam ItemT Type of an item at the index corresponding to this ID
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
 * it does not (a "null" ID, analogous to a null pointer: it does not
 * correspond to a valid value). A "true" ID will always compare less than a
 * "false" ID: you can use \c std::partition and \c erase to remove invalid IDs
 * from a vector.
 *
 * See also \c id_cast below for checked construction of OpaqueIds from generic
 * integer values (avoid compile-time warnings or errors from signed/truncated
 * integers).
 *
 * \todo This interface will be changed to be more like \c std::optional : \c
 * size_type will become \c value_type (the value of a 'dereferenced' ID) and
 * \c operator* or \c value will be used to access the integer.
 */
template<class ItemT, class SizeT = ::celeritas::size_type>
class OpaqueId
{
    static_assert(std::is_unsigned_v<SizeT> && !std::is_same_v<SizeT, bool>,
                  "SizeT must be unsigned.");

  public:
    //!@{
    //! \name Type aliases
    using value_type [[deprecated]] = ItemT;
    using size_type = SizeT;
    //!@}

  public:
    //! Default to null state
    CELER_CONSTEXPR_FUNCTION OpaqueId() : value_(nullid_val_) {}

    //! Construct explicitly with stored value
    explicit CELER_CONSTEXPR_FUNCTION OpaqueId(size_type index) : value_(index)
    {
    }

    //! Whether this ID is in a valid (assigned) state
    explicit CELER_CONSTEXPR_FUNCTION operator bool() const
    {
        return value_ != nullid_val_;
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

    //! Value indicating the ID is not assigned
    static constexpr size_type nullid_val_ = detail::nullid_value<size_type>;
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
 * \note The value cannot be the underlying "null" value; i.e.
 * <code> static_cast<FooId>(FooId{}.unchecked_get()) </code> will not work.
 */
template<class IdT, class U>
inline CELER_FUNCTION IdT id_cast(U value) noexcept(!CELERITAS_DEBUG)
{
    static_assert(std::is_integral_v<U>);
    static_assert(std::is_integral_v<typename IdT::size_type>);

    return IdT{detail::id_cast_impl<typename IdT::size_type, U>(value)};
}

//---------------------------------------------------------------------------//
#define CELER_DEFINE_OPAQUEID_CMP(TOKEN)                             \
    template<class I, class T>                                       \
    CELER_CONSTEXPR_FUNCTION bool operator TOKEN(OpaqueId<I, T> lhs, \
                                                 OpaqueId<I, T> rhs) \
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
template<class I, class T, class U>
CELER_CONSTEXPR_FUNCTION bool operator<(OpaqueId<I, T> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) < rhs);
}

//---------------------------------------------------------------------------//
//! Allow less-than-equal comparison with *integer* for container comparison
template<class I, class T, class U>
CELER_CONSTEXPR_FUNCTION bool operator<=(OpaqueId<I, T> lhs, U rhs)
{
    // Cast to RHS
    return lhs && (U(lhs.unchecked_get()) <= rhs);
}

//---------------------------------------------------------------------------//
//! Get the distance between two opaque IDs
template<class I, class T>
inline CELER_FUNCTION T operator-(OpaqueId<I, T> self, OpaqueId<I, T> other)
{
    CELER_EXPECT(self);
    CELER_EXPECT(other);
    return self.unchecked_get() - other.unchecked_get();
}

//---------------------------------------------------------------------------//
//! Increment an opaque ID by an offset
template<class I, class T>
inline CELER_FUNCTION OpaqueId<I, T>
operator+(OpaqueId<I, T> id, std::make_signed_t<T> offset)
{
    CELER_EXPECT(id);
    CELER_EXPECT(offset >= 0 || static_cast<T>(-offset) <= id.unchecked_get());
    // Note: an extra cast is needed for short T due to integer promotion
    return OpaqueId<I, T>{
        static_cast<T>(id.unchecked_get() + static_cast<T>(offset))};
}

//---------------------------------------------------------------------------//
//! Decrement an opaque ID by an offset
template<class I, class T>
inline CELER_FUNCTION OpaqueId<I, T>
operator-(OpaqueId<I, T> id, std::make_signed_t<T> offset)
{
    CELER_EXPECT(id);
    CELER_EXPECT(offset <= 0 || static_cast<T>(offset) <= id.unchecked_get());
    // Note: an extra cast is needed for short T due to integer promotion
    return OpaqueId<I, T>{
        static_cast<T>(id.unchecked_get() - static_cast<T>(offset))};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//! Specialization for std::hash for unordered storage.
template<class I, class T>
struct hash<celeritas::OpaqueId<I, T>>
{
    std::size_t operator()(celeritas::OpaqueId<I, T> const& id) const noexcept
    {
        return std::hash<T>()(id.unchecked_get());
    }
};
}  // namespace std
//! \endcond
