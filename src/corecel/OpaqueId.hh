//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/OpaqueId.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "Assert.hh"
#include "Macros.hh"
#include "Types.hh"

#if !CELER_DEVICE_COMPILE
#    include <functional>
#    include <ostream>
#endif

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! Sentinel value for an unassigned opaque ID
template<class T>
inline constexpr T nullid_value{static_cast<T>(-1)};
//---------------------------------------------------------------------------//
}  // namespace detail

//! Tag type used for \c nullid
struct nullid_t
{
};

//! Tag instance used to instantiate and compare to a null OpaqueId
inline constexpr nullid_t nullid;

//---------------------------------------------------------------------------//
/*!
 * Type-safe index for accessing an array or collection of data.
 *
 * \tparam ItemT Type of an item at the index corresponding to this ID
 * \tparam SizeT Unsigned integer index
 *
 * Indexing into arrays with integers, rather than storing pointers, is
 * \em key to easy and safe data management across host/device boundaries.
 * Pointers in C++ can act as a reference to an array or element of data, and
 * they also have a \em type, which not only gives the stride width in bytes
 * but also <em>prevents accidental aliasing</em>.
 *
 * The \c OpaqueId class is an attempt to model integer indexing
 * (device-friendly) with pointer semantics (type-safe).
 * Annotating index offsets with a type gives the offsets a semantic meaning,
 * and it gives the developer compile-time type safety.
 * As an example, it prevents index arguments in a function call from being
 * provided out of order.
 *
 * In addition to representing an offset and type, this can also model a null
 * pointer: an \c OpaqueId object evaluates to \c true if it has a value
 * (`OpaqueId{3}`), or \c false if it does not (`OpaqueId{}`).
 * The invalid state is usually referred to in the codebase as a "null ID".
 *
 * The class is roughly modeled after \c std::optional<SizeT> (but efficient
 * as it has no extra boolean flag thanks to the use of a sentinel value).
 * The default-constructed value, \c nullid, cannot be used to index into an
 * array, nor does it represent a valid element.
 *
 * \tip A valid ID will always compare less than a null ID: you can use
 *      \c std::partition and \c erase to remove null IDs from a vector.
 *
 * \par Synopsis
 *
 * A default-constructed OpaqueId is "null". It can be constructed explicitly
 * from unsigned integers. (Use \c id_cast for safe construction from integer
 * or differently-sized values.)
 *
 * Usage:
 * - Index into \c Collection objects
 * - Check for nullity with \c bool, by comparing with \c nullid,
 * - Access with \c .value() or \c operator*
 *
 * The OpaqueId is hashable, sortable, and printable. It can be loaded via
 * texture-backed device memory using \c ldg .
 *
 * \par Related helper functions and types
 * - \c nullid is an instance of \c nullid_t that compares to any OpaqueId as
 *   its "null" value.
 * - \c is_opaque_id_v allows checking for generic types
 * - \c id_size_t is a descriptive alias to get the unsigned integer \c
 *   value_type of an opaque ID, used for capacities.
 * - \c id_cast safely converts integers to OpaqueId .
 *
 * \par About the ItemT tag
 * If this class is used for indexing into an array, then \c ValueT argument
 * should usually be the value type of the array:
 * <code>FooRecord operator[](OpaqueId<FooRecord>)</code>
 * Otherwise, the convention is to use an anonymous <code>struct Bar_</code> to
 * tag the ID type.
 *
 * \note Comparators are defined as inline friend functions to allow
 * ADL-assisted conversion, including from \c LdgWrapper.
 *
 */
template<class ItemT, class SizeT = ::celeritas::size_type>
class OpaqueId
{
    static_assert(std::is_unsigned_v<SizeT> && !std::is_same_v<SizeT, bool>,
                  "SizeT must be unsigned.");

    static constexpr bool ndebug = !CELERITAS_DEBUG;

  public:
    //!@{
    //! \name Type aliases
    using tag_type = ItemT;
    using value_type = SizeT;
    using size_type = value_type;  // DEPRECATED
    //!@}

  public:
    //! Construct implicitly from a null type
    CELER_CEF OpaqueId(nullid_t) : value_(null_) {}

    //! Default to null state
    CELER_CEF OpaqueId() : OpaqueId(nullid) {}

    //! Construct explicitly with a stored value
    explicit CELER_CEF OpaqueId(value_type index) : value_(index) {}

    //! Whether this ID is in a valid (assigned) state
    explicit CELER_CEF operator bool() const noexcept
    {
        return value_ != null_;
    }

    //! Dereference to access the value
    CELER_CEF const value_type& operator*() const& noexcept(ndebug)
    {
        CELER_EXPECT(*this);
        return value_;
    }

    //!@{
    //! \name Deprecated modification
    //! \deprecated Remove in v1.0

    //! Pre-increment of the ID
    CELER_CEF OpaqueId& operator++() noexcept(ndebug)
    {
        CELER_EXPECT(*this);
        value_ += 1;
        return *this;
    }

    //! Post-increment of the ID
    CELER_CEF OpaqueId operator++(int) noexcept(ndebug)
    {
        OpaqueId old{*this};
        ++*this;
        return old;
    }

    //! Pre-decrement of the ID
    CELER_CEF OpaqueId& operator--() noexcept(ndebug)
    {
        CELER_EXPECT(*this && value_ > 0);
        value_ -= 1;
        return *this;
    }

    //! Post-decrement of the ID
    CELER_CEF OpaqueId operator--(int) noexcept(ndebug)
    {
        OpaqueId old{*this};
        --*this;
        return old;
    }

    //!@}

    //!@{
    //! \name Deprecated access
    //! \deprecated Remove in v1.0

    //! Get the ID's value
    CELER_FIF value_type get() const noexcept(ndebug)
    {
        CELER_EXPECT(*this);
        return value_;
    }

    //! Get the value without checking for validity (atypical)
    CELER_CEF value_type unchecked_get() const noexcept { return value_; }

    //! Access the underlying data for more efficient loading on device
    CELER_CEF value_type const* data() const noexcept { return &value_; }

    //!@}

    //// INLINE COMPARATOR FRIENDS ////

#define CELER_DEFINE_OPAQUEID_CMP(TOKEN)                                      \
    CELER_CEF friend bool operator TOKEN(OpaqueId lhs, OpaqueId rhs) noexcept \
    {                                                                         \
        return lhs.unchecked_get() TOKEN rhs.unchecked_get();                 \
    }

    //!@{
    //! Compare two OpaqueId of the same type
    CELER_DEFINE_OPAQUEID_CMP(==)
    CELER_DEFINE_OPAQUEID_CMP(!=)
    CELER_DEFINE_OPAQUEID_CMP(<)
    CELER_DEFINE_OPAQUEID_CMP(>)
    CELER_DEFINE_OPAQUEID_CMP(<=)
    CELER_DEFINE_OPAQUEID_CMP(>=)
    //!@}

#undef CELER_DEFINE_OPAQUEID_CMP
#define CELER_DEFINE_OPAQUEID_CMP(TOKEN)                               \
    template<class U>                                                  \
    CELER_CEF friend auto operator TOKEN(OpaqueId lhs, U rhs) noexcept \
        -> std::enable_if_t<std::is_unsigned_v<U>, bool>               \
    {                                                                  \
        return lhs && (static_cast<U>(lhs.unchecked_get()) TOKEN rhs); \
    }

    //!@{
    //! Allow less-than comparison with unsigned int for containers
    CELER_DEFINE_OPAQUEID_CMP(<)
    CELER_DEFINE_OPAQUEID_CMP(<=)
    //!@}

#undef CELER_DEFINE_OPAQUEID_CMP

    //// INLINE OPERATOR FRIENDS ////

    //! Get the distance between two opaque IDs
    CELER_FUNCTION friend SizeT operator-(OpaqueId self, OpaqueId other)
    {
        CELER_EXPECT(self);
        CELER_EXPECT(other);
        return self.unchecked_get() - other.unchecked_get();
    }

    //! Increment an opaque ID by an offset, checking against underflow
    template<class U>
    CELER_FUNCTION friend auto operator+(OpaqueId id, U offset)
        -> std::enable_if_t<std::is_integral_v<U>, OpaqueId>
    {
        CELER_EXPECT(id);
        CELER_EXPECT(OpaqueId::is_safe_offset(id.unchecked_get(), offset));

        // Note: an extra cast is needed for short SizeT due to integer
        // promotion
        return OpaqueId{static_cast<SizeT>(id.unchecked_get() + offset)};
    }

    //! Increment an opaque ID by an offset (symmetric)
    template<class U>
    CELER_FUNCTION friend auto operator+(U offset, OpaqueId id)
        -> std::enable_if_t<std::is_integral_v<U>, OpaqueId>
    {
        return id + offset;
    }

    //! Decrement an opaque ID by an offset
    template<class U>
    CELER_FUNCTION friend auto operator-(OpaqueId id, U offset)
        -> std::enable_if_t<std::is_integral_v<U>, OpaqueId>
    {
        CELER_EXPECT(id);
        CELER_EXPECT(offset <= 0
                     || static_cast<SizeT>(offset) <= id.unchecked_get());
        // Note: an extra cast is needed for short SizeT due to integer
        // promotion
        return OpaqueId{static_cast<SizeT>(id.unchecked_get()
                                           - static_cast<SizeT>(offset))};
    }

  private:
    size_type value_;

    //! Value indicating the ID is not assigned
    static constexpr size_type null_ = detail::nullid_value<size_type>;

    //// HELPER FUNCTIONS ////

    template<class U>
    static CELER_CEF bool is_safe_offset(SizeT value, U offset)
    {
        if constexpr (std::is_unsigned_v<U>)
        {
            return true;
        }
        else
        {
            if (offset >= 0)
            {
                // NOTE: we do not check for overflow
                return true;
            }
            return static_cast<SizeT>(U{0} - offset) <= value;
        }
    }
};

//---------------------------------------------------------------------------//
// DETAIL IMPLEMENTATION
// (not a separate file due to living in the top level)
//---------------------------------------------------------------------------//

namespace detail
{
//---------------------------------------------------------------------------//
//! Safely cast from one integer T to another U, avoiding the sentinel value
template<class T, class U>
inline CELER_FUNCTION T id_cast_impl(U value) noexcept(!CELERITAS_DEBUG)
{
    constexpr auto null_val = detail::nullid_value<T>;

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
        CELER_EXPECT(static_cast<C>(value) < static_cast<C>(null_val));
    }
    else
    {
        // Check that value is *not* the null value
        CELER_EXPECT(static_cast<T>(value) != null_val);
    }

    return static_cast<T>(value);
}

//---------------------------------------------------------------------------//
//! Template matching to determine if T is an OpaqueId
template<class T>
struct IsOpaqueId : std::false_type
{
};

template<class V, class S>
struct IsOpaqueId<OpaqueId<V, S>> : std::true_type
{
};

template<class V, class S>
struct IsOpaqueId<OpaqueId<V, S> const> : std::true_type
{
};

#if !CELER_DEVICE_COMPILE
// Print an opaque ID: ignore instantiator to reduce duplicate symbols
template<class S>
inline void stream_opaqueid_impl(std::ostream& os, S v, S nullid)
{
    os << '{';
    if (v != nullid)
    {
        os << v;
    }
    os << '}';
}

// Specialization avoids printing integers as '\x1'
template<>
CELER_FORCEINLINE void
stream_opaqueid_impl(std::ostream& os, unsigned char v, unsigned char nullid)
{
    return stream_opaqueid_impl(
        os, static_cast<unsigned int>(v), static_cast<unsigned int>(nullid));
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
// FREE IMPLEMENTATIONS
//---------------------------------------------------------------------------//

//! True if T is an OpaqueID
template<class T>
inline constexpr bool is_opaque_id_v = detail::IsOpaqueId<T>::value;

//! Get the unsigned integer corresponding to the ID's capacity
template<class T>
using id_size_type
    = std::conditional_t<is_opaque_id_v<T>, typename T::value_type, void>;

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
inline CELER_FUNCTION auto id_cast(U value) noexcept(!CELERITAS_DEBUG)
    -> std::enable_if_t<is_opaque_id_v<IdT> && std::is_integral_v<U>, IdT>
{
    return IdT{detail::id_cast_impl<typename IdT::size_type, U>(value)};
}

//---------------------------------------------------------------------------//
/*!
 * Support loading OpaqueId via GPU cache.
 */
template<class I, class T>
CELER_CEF T const* ldg_data(OpaqueId<I, T> const* ptr) noexcept
{
    return ptr->data();
}

#if !CELER_DEVICE_COMPILE
//---------------------------------------------------------------------------//
/*!
 * Output an opaque ID's value or a placeholder if unavailable.
 */
template<class V, class S>
CELER_FORCEINLINE std::ostream&
operator<<(std::ostream& os, OpaqueId<V, S> const& v)
{
    detail::stream_opaqueid_impl(os, *v.data(), detail::nullid_value<S>);
    return os;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas

#if !CELER_DEVICE_COMPILE
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
#endif
