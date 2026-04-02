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

#if !CELER_DEVICE_COMPILE
#    include <ostream>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Sentinel value for an unassigned opaque ID
template<class T>
inline constexpr T nullid_value{static_cast<T>(-1)};

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
 * \note Comparators are defined as inline friend functions to allow
 * ADL-assisted conversion, including from \c LdgWrapper (see \ref ldg).
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
    CELER_CEF OpaqueId() : value_(null_) {}

    //! Construct explicitly with stored value
    explicit CELER_CEF OpaqueId(size_type index) : value_(index) {}

    //! Whether this ID is in a valid (assigned) state
    explicit CELER_CEF operator bool() const { return value_ != null_; }

    //! Pre-increment of the ID
    CELER_CEF OpaqueId& operator++()
    {
        CELER_EXPECT(*this);
        value_ += 1;
        return *this;
    }

    //! Post-increment of the ID
    CELER_CEF OpaqueId operator++(int)
    {
        OpaqueId old{*this};
        ++*this;
        return old;
    }

    //! Pre-decrement of the ID
    CELER_CEF OpaqueId& operator--()
    {
        CELER_EXPECT(*this && value_ > 0);
        value_ -= 1;
        return *this;
    }

    //! Post-decrement of the ID
    CELER_CEF OpaqueId operator--(int)
    {
        OpaqueId old{*this};
        --*this;
        return old;
    }

    //! Get the ID's value
    CELER_CEF size_type get() const
    {
        CELER_EXPECT(*this);
        return value_;
    }

    //! Get the value without checking for validity (atypical)
    CELER_CONSTEXPR_FUNCTION size_type unchecked_get() const { return value_; }

    //! Access the underlying data for more efficient loading from memory
    CELER_CONSTEXPR_FUNCTION size_type const* data() const { return &value_; }

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
    static constexpr size_type null_ = nullid_value<size_type>;

    //// HELPER FUNCTIONS ////

    template<class U>
    static CELER_CONSTEXPR_FUNCTION bool is_safe_offset(SizeT value, U offset)
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

#if !CELER_DEVICE_COMPILE
//---------------------------------------------------------------------------//
/*!
 * Output an opaque ID's value or a placeholder if unavailable.
 */
template<class V, class S>
CELER_FORCEINLINE std::ostream&
operator<<(std::ostream& os, OpaqueId<V, S> const& v)
{
    detail::stream_opaqueid_impl(os, v.unchecked_get(), nullid_value<S>);
    return os;
}
#endif

//---------------------------------------------------------------------------//
// FREE FUNCTION LDG SUPPORT
//---------------------------------------------------------------------------//

//! Cached const global loading support for OpaqueId
template<class I, class T>
CELER_CONSTEXPR_FUNCTION T const* ldg_data(OpaqueId<I, T> const* ptr) noexcept
{
    return ptr->data();
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
