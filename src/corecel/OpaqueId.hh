//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/OpaqueId.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/data/Ldg.hh"

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
template<class I>
inline constexpr I nullid_value{static_cast<I>(-1)};

#if !CELER_DEVICE_COMPILE
template<class I>
void stream_opaqueid_impl(std::ostream&, I, I);
#endif

//---------------------------------------------------------------------------//
}  // namespace detail

//! Tag type used for \c nullid
struct nullid_t
{
    constexpr explicit nullid_t(int) {}
};

//! Tag instance used to instantiate and compare to a null OpaqueId
inline constexpr nullid_t nullid{0};

//---------------------------------------------------------------------------//
/*!
 * Type-safe "optional" index for accessing an array or collection of data.
 *
 * \tparam TagT Type of an item at the index corresponding to this ID
 * \tparam IndexT Unsigned integer acting as the stored value
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
 * The typical usage of an OpaqueId should be as \c std::optional<IndexT>.
 * The default-constructed value, \c nullid, cannot be used to index into an
 * array, nor does it represent a valid element.
 * An \c OpaqueId object evaluates to \c true if it has a value
 * (`OpaqueId{3}`), or \c false if it does not (`OpaqueId{}`).
 * The invalid state is usually referred to in the codebase as a "null ID".
 * Analogous to \c std::optional, \c nullid can be used for comparison,
 * assignment, and construction.
 *
 * \par Construction
 * - Default: result is \c nullid
 * - Implicitly from \c nullid
 * - Explicitly from a compatible unsigned integer
 * - Via \c id_cast for safe construction from general (or differently sized)
 *   integers
 *
 * \par Usage
 * - Check for nullity with \c bool or by comparing with \c nullid
 * - Check for validity as a container index with
 *   <code>id &lt; vec.size()</code>
 * - Access value with \c operator* : <code>vec[*id]</code>
 * - Access data with \c Collection::operator[]
 * - Loop over consecutive IDs with \c range
 * - Pre- and post- increment and decrement
 * - Subtract two IDs to get a difference
 *
 * The OpaqueId is hashable, sortable, and printable.
 * It can be loaded via cached device memory using \c ldg .
 *
 * \note A valid ID will always compare less than a null ID: you can use
 *      \c std::partition and \c erase to remove null IDs from a vector.
 *
 * \note Comparators are defined as inline friend functions to allow
 * ADL-assisted conversion, including from \c LdgWrapper.
 *
 * \par Related helper functions and types
 * - \c nullid is an instance of \c nullid_t that compares to any OpaqueId as
 *   its "null" value.
 * - \c is_opaque_id_v allows checking for generic types
 * - \c MakeSize_t is a descriptive type alias to get the unsigned integer
 *   \c value_type of an opaque ID, used for container capacities.
 * - \c id_cast safely converts integers to OpaqueId.
 *
 * \par About the TagT template parameter
 * If this class is used for indexing into an array, then \c TagT argument
 * should usually be the value type of the array:
 * \code
 * FooRecord operator[](OpaqueId<FooRecord>);
 * \endcode
 * Otherwise, the convention is to use an anonymous tag:
 * \code
  using FooId = OpaqueId<struct Foo_>;
 * \endcode
 */
template<class TagT, class IndexT = ::celeritas::size_type>
class OpaqueId
{
    static_assert(std::is_unsigned_v<IndexT> && !std::is_same_v<IndexT, bool>,
                  "IndexT must be unsigned.");

    static constexpr bool ndebug = !CELERITAS_DEBUG;

  public:
    //!@{
    //! \name Type aliases
    using tag_type = TagT;
    using value_type = IndexT;  // like std::optional
    using index_type = IndexT;
    using difference_type = std::make_signed_t<IndexT>;
    using size_type = value_type;  // DEPRECATED
    //!@}

  public:
    //! Construct implicitly from a null type
    CELER_CEF OpaqueId(nullid_t) : value_(null_) {}

    //! Default to null state
    CELER_CEF OpaqueId() : OpaqueId(nullid) {}

    //! Construct explicitly with a stored value (validity *not* checked)
    explicit CELER_CEF OpaqueId(value_type index) : value_(index) {}

    //! Whether this ID is in a valid (assigned) state
    explicit CELER_CEF operator bool() const noexcept
    {
        return value_ != null_;
    }

    //! Unchecked dereference: caller must ensure the ID is valid
    CELER_CEF value_type operator*() const noexcept { return value_; }

    //! Get the value, asserting non-null in debug builds
    CELER_CEF value_type value() const noexcept(ndebug)
    {
        CELER_EXPECT(*this);
        return value_;
    }

    //! Access the underlying data for cached loading on device
    CELER_CEF value_type const* data() const noexcept { return &value_; }

    //!@{
    //! \name Pointer-like operators
    //! \note Incrementing/decrementing to/from null is prohibited

    CELER_CEF OpaqueId& operator++() noexcept(ndebug)
    {
        CELER_EXPECT(value_ < (null_ - 1));
        value_ += 1;
        return *this;
    }

    CELER_CEF OpaqueId operator++(int) noexcept(ndebug)
    {
        OpaqueId old{*this};
        ++*this;
        return old;
    }

    CELER_CEF OpaqueId& operator--() noexcept(ndebug)
    {
        CELER_EXPECT(*this && value_ > 0);
        value_ -= 1;
        return *this;
    }

    CELER_CEF OpaqueId operator--(int) noexcept(ndebug)
    {
        OpaqueId old{*this};
        --*this;
        return old;
    }

    //!@}

    //!@{
    //! \name Pointer-like arithmetic
    //! Get the distance between two opaque IDs
    CELER_FUNCTION friend difference_type
    operator-(OpaqueId self, OpaqueId other)
    {
        CELER_EXPECT(self);
        CELER_EXPECT(other);
        return static_cast<difference_type>(*self - *other);
    }

    //! Increment an opaque ID by an offset, checking against underflow
    template<class J>
    CELER_FUNCTION friend auto operator+(OpaqueId id, J offset)
        -> std::enable_if_t<std::is_integral_v<J>, OpaqueId>
    {
        CELER_EXPECT(id);
        CELER_EXPECT(OpaqueId::is_safe_offset(*id, offset));

        // Note: an extra cast is needed for short IndexT with long J
        return OpaqueId{static_cast<IndexT>(*id + offset)};
    }

    //! Increment an opaque ID by an offset (symmetric)
    template<class J>
    CELER_FUNCTION friend auto operator+(J offset, OpaqueId id)
        -> std::enable_if_t<std::is_integral_v<J>, OpaqueId>
    {
        return id + offset;
    }

    //! Decrement an opaque ID by an offset
    template<class J>
    CELER_FUNCTION friend auto operator-(OpaqueId id, J offset)
        -> std::enable_if_t<std::is_integral_v<J>, OpaqueId>
    {
        CELER_EXPECT(id);
        CELER_EXPECT(offset <= 0 || static_cast<IndexT>(offset) <= *id);
        // Note: an extra cast is needed for short I with long J
        return OpaqueId{static_cast<IndexT>(*id - static_cast<IndexT>(offset))};
    }
    //!@}

    //!@{
    //! \name Deprecated access
    //! \deprecated Remove in v1.0

    //! Get the ID's value: use \c value() instead
    CELER_CEF value_type get() const noexcept(ndebug) { return this->value(); }

    //! Get the value without checking: use \c operator* instead
    CELER_CEF value_type unchecked_get() const noexcept { return value_; }

    //!@}

    //// TEMPLATE FRIEND OPERATORS ////

#define CELER_DEFINE_OPAQUEID_CMP(TOKEN)                                      \
    CELER_CEF friend bool operator TOKEN(OpaqueId lhs, OpaqueId rhs) noexcept \
    {                                                                         \
        return *lhs TOKEN * rhs;                                              \
    }

    //!@{
    //! \name Compare two OpaqueId of the same type
    CELER_DEFINE_OPAQUEID_CMP(==)
    CELER_DEFINE_OPAQUEID_CMP(!=)
    CELER_DEFINE_OPAQUEID_CMP(<)
    CELER_DEFINE_OPAQUEID_CMP(>)
    CELER_DEFINE_OPAQUEID_CMP(<=)
    CELER_DEFINE_OPAQUEID_CMP(>=)
    //!@}

    //!@{
    //! \name Compare with nullid
    CELER_CEF friend bool operator==(OpaqueId id, nullid_t) noexcept
    {
        return !id;
    }
    CELER_CEF friend bool operator==(nullid_t, OpaqueId id) noexcept
    {
        return !id;
    }
    CELER_CEF friend bool operator!=(OpaqueId id, nullid_t) noexcept
    {
        return static_cast<bool>(id);
    }
    CELER_CEF friend bool operator!=(nullid_t, OpaqueId id) noexcept
    {
        return static_cast<bool>(id);
    }
    //!@}

    //! Allow loading via \c ldg
    CELER_CONSTEXPR_FUNCTION friend OpaqueId ldg(OpaqueId const* id)
    {
        using ::celeritas::ldg;
        return OpaqueId{ldg(&id->value_)};
    }

#undef CELER_DEFINE_OPAQUEID_CMP
#define CELER_DEFINE_OPAQUEID_CMP(TOKEN)                               \
    template<class J>                                                  \
    CELER_CEF friend auto operator TOKEN(OpaqueId lhs, J rhs) noexcept \
        -> std::enable_if_t<std::is_unsigned_v<J>, bool>               \
    {                                                                  \
        return lhs && (static_cast<J>(*lhs) TOKEN rhs);                \
    }

    //!@{
    //! \name Compare with unsigned int
    //! This allows size checking for containers
    CELER_DEFINE_OPAQUEID_CMP(<)
    CELER_DEFINE_OPAQUEID_CMP(<=)
    //!@}

#undef CELER_DEFINE_OPAQUEID_CMP

#if !CELER_DEVICE_COMPILE
    //! Output an opaque ID's value or a placeholder if unavailable.
    CELER_FORCEINLINE friend std::ostream&
    operator<<(std::ostream& os, OpaqueId v)
    {
        detail::stream_opaqueid_impl(os, v.value_, v.null_);
        return os;
    }
#endif

  private:
    //// DATA ////

    value_type value_;

    //! Value indicating the ID is not assigned
    static constexpr value_type null_ = detail::nullid_value<value_type>;

    //// HELPER FUNCTIONS ////

    template<class J>
    static CELER_CEF bool is_safe_offset(IndexT value, J offset)
    {
        if constexpr (std::is_unsigned_v<J>)
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
            return static_cast<IndexT>(J{0} - offset) <= value;
        }
    }
};

//! Specialize size type for opaque IDs
template<class I, class V>
struct MakeSize<OpaqueId<I, V>>
{
    using type = V;
};

//! Automatically load read-only OpaqueIDs with LDG
template<class I, class V>
struct IsAutoLdg<OpaqueId<I, V>> : std::true_type
{
};

//---------------------------------------------------------------------------//
// DETAIL IMPLEMENTATION
// (not a separate file due to living in the top level)
//---------------------------------------------------------------------------//

namespace detail
{
//---------------------------------------------------------------------------//
//! Safely cast from one integer T to another J, avoiding the sentinel value
template<class T, class J>
inline CELER_FUNCTION T id_cast_impl(J value) noexcept(!CELERITAS_DEBUG)
{
    constexpr auto null_val = detail::nullid_value<T>;

    if constexpr (std::is_signed_v<J>)
    {
        CELER_EXPECT(value >= 0);
    }

    if constexpr (!std::is_same_v<T, J>)
    {
        // Check that the cast value is within the integer range [0, N-1)
        using C = std::common_type_t<T, std::make_unsigned_t<J>>;
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

template<class I, class V>
struct IsOpaqueId<OpaqueId<I, V>> : std::true_type
{
};

#if !CELER_DEVICE_COMPILE
// Print an opaque ID: ignore instantiator to reduce duplicate symbols
template<class V>
inline void stream_opaqueid_impl(std::ostream& os, V v, V nullint)
{
    os << '{';
    if (v != nullint)
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
inline constexpr bool is_opaque_id_v
    = detail::IsOpaqueId<std::remove_cv_t<T>>::value;

//---------------------------------------------------------------------------//
/*!
 * Safely create an OpaqueId from an integer of any type.
 *
 * This asserts that the integer is in the \em valid range of the target ID
 * type, and casts to it.
 *
 * \note The value cannot be the underlying "null" value; i.e.
 * <code> static_cast<FooId>(*FooId{}) </code> will not work.
 */
template<class IdT, class J>
inline CELER_FUNCTION auto id_cast(J value) noexcept(!CELERITAS_DEBUG)
    -> std::enable_if_t<is_opaque_id_v<IdT> && std::is_integral_v<J>, IdT>
{
    return IdT{detail::id_cast_impl<typename IdT::value_type, J>(value)};
}

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
        return std::hash<T>()(*id);
    }
};
}  // namespace std
//! \endcond
#endif
