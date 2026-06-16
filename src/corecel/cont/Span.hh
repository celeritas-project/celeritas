//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Span.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "corecel/Macros.hh"

#include "detail/SpanImpl.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/StreamableContainer.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Sentinel value for span of dynamic type
constexpr std::size_t dynamic_extent = detail::dynamic_extent;

//---------------------------------------------------------------------------//
// FORWARD DECLARATIONS
//---------------------------------------------------------------------------//
template<class T, std::size_t N>
class Array;

//---------------------------------------------------------------------------//
/*!
 * Non-owning device-compatible reference to a contiguous span of data.
 * \tparam T value type
 * \tparam Extent fixed size; defaults to dynamic.
 *
 * A \c Span, like \c std::string_view, provides access to externally managed
 * data. In Celeritas, this class is typically used as a return result
 * from accessing a range of elements in a \c Collection.
 *
 * This implementation is a \em nonconforming backport of the C++20 \c
 * std::span. Improvements for standards compatibility are welcome as long as
 * they retain the same behavior in device code. Important differences from
 * the standard \c std::span include:
 * - Supports a special marker/tag type \c LdgValue<T> which causes element
 *   accessors and iterators to use value-semantics loads (optimized device
 *   loads) instead of references.
 * - Uses a restricted constructor for iterators: instead of two separate
 *   iterator/end types, it uses only one.
 * - Provides additional free helpers tailored to Celeritas: \c make_span
 *   overloads for \c Array<T,N>, C arrays, and generic containers, plus
 *   \c to_array() convenience and a host-only \c operator<< using
 *   \c StreamableContainer.
 * - All public methods are decorated with \c CELER_CONSTEXPR_FUNCTION for
 *   host/device compatibility.
 * - Some subview helpers use \c CELER_EXPECT to check for bounds validation in
 *   debug builds.
 * - Dynamic-to-fixed conversion performs runtime checks when \c
 *   CELERITAS_DEBUG is on.
 *
 * \par Construction
 *
 * - Default: empty span
 * - Implicit: pointer and size
 * - Implicit: two \c contiguous_iterator \c (first,last)
 * - Implicit: C arrays and celeritas::Array when extents are compatible
 * - Implicit: fixed-to-dynamic Span
 *
 * \par Data access
 *
 * - Element access: \c operator[], \c front(), \c back()
 * - Observers: \c data(), \c size(), \c size_bytes(), \c empty()
 * - Iteration: \c begin(), \c end()
 *
 * \par Subviews and utilities
 *
 * - \c first<Count>(), \c first(count), \c last<Count>(), \c last(count)
 * - \c subspan<Offset,Count>() and \c subspan(offset,count) for compile-time
 *   and runtime subviews
 * - Deduction guides for pointer+size, iterator pairs, C arrays, and Array
 * - Free functions \c make_span(...) and \c to_array(...)
 */
template<class T, std::size_t Extent = dynamic_extent>
class Span
{
    using SpanTraitsT = detail::SpanTraits<T>;
    static constexpr bool ndebug = !CELERITAS_DEBUG;
    static constexpr bool ndebug_or_dyn = (ndebug || Extent == dynamic_extent);

  public:
    //!@{
    //! \name Type aliases
    using element_type = typename SpanTraitsT::element_type;
    using value_type = std::remove_cv_t<element_type>;
    using size_type = std::size_t;
    using pointer = typename SpanTraitsT::pointer;
    using const_pointer = typename SpanTraitsT::const_pointer;
    using reference = typename SpanTraitsT::reference;
    using const_reference = typename SpanTraitsT::const_reference;
    using iterator = typename SpanTraitsT::iterator;
    using const_iterator = typename SpanTraitsT::const_iterator;
    //!@}

    //! Size (may be dynamic)
    static constexpr std::size_t extent = Extent;

  public:
    //// CONSTRUCTION ////

    //! Construct with default null pointer and size zero
    constexpr Span() = default;

    //! Construct from data and size
    CELER_CONSTEXPR_FUNCTION
    Span(pointer d, std::size_t s) noexcept(ndebug_or_dyn) : s_(d, s) {}

    /*!
     * Construct from two contiguous random-access iterators.
     *
     * \note If \c Extent is fixed-size, the \c SpanImpl will fire an assertion
     * in debug mode if the size does not match.
     * \todo should be explicit unless dynamic size (requires C++20)
     */
    template<class Iter>
    CELER_CONSTEXPR_FUNCTION Span(Iter first, Iter last) noexcept(ndebug_or_dyn)
        : s_(first == last ? nullptr : &(*first),
             static_cast<std::size_t>(last - first))
    {
        CELER_EXPECT(last >= first);
    }

    //! Construct from a C array
    template<std::size_t N,
             std::enable_if_t<N == Extent || Extent == dynamic_extent, bool> = true>
    CELER_CONSTEXPR_FUNCTION Span(element_type (&arr)[N]) noexcept : s_(arr, N)
    {
    }

    /*!
     * Construct implicitly from a mutable \c Array.
     *
     * Enabled only when \c N matches \c Extent exactly or \c Extent is
     * \c dynamic_extent, so the conversion is always statically safe and
     * can be implicit unconditionally.
     */
    template<class U,
             std::size_t N,
             std::enable_if_t<detail::is_array_convertible_v<U, T>
                                  && (N == Extent || Extent == dynamic_extent),
                              bool> = true>
    CELER_CONSTEXPR_FUNCTION Span(Array<U, N>& arr) noexcept
        : s_(arr.data(), N)
    {
    }

    /*!
     * Construct implicitly from a const \c Array.
     *
     * Enabled only when \c N matches \c Extent exactly or \c Extent is
     * \c dynamic_extent, so the conversion is always statically safe and
     * can be implicit unconditionally.
     */
    template<class U,
             std::size_t N,
             std::enable_if_t<detail::is_array_convertible_v<U const, T>
                                  && (N == Extent || Extent == dynamic_extent),
                              bool> = true>
    CELER_CONSTEXPR_FUNCTION Span(Array<U, N> const& arr) noexcept
        : s_(arr.data(), N)
    {
    }

    /*!
     * Construct \c implicitly from convertible span and extent.
     *
     * Note that the enable-if prevents LdgSpan->Span conversion.
     * Conversions that may require a runtime size check (dynamic-extent
     * source to fixed-extent destination) are made explicit to avoid
     * accidental UB, matching std::span's behavior.
     *
     * Implicit conversion for cases that are statically safe (e.g.,
     * fixed->dynamic, fixed->same-fixed, or just element-type
     * qualification changes).
     */
    template<class U,
             std::size_t E2,
             std::enable_if_t<detail::is_array_convertible_v<U, T>
                                  && (E2 == Extent || Extent == dynamic_extent),
                              bool> = true>
    CELER_CONSTEXPR_FUNCTION
    Span(Span<U, E2> const& other) noexcept(ndebug_or_dyn)
        : s_(other.data(), other.size())
    {
    }

    /*!
     * Require \em explicit conversion from dynamic to fixed extent.
     *
     * Runtime size compatibility is checked in \c detail::SpanImpl.
     */
    template<class U,
             std::size_t E2,
             std::enable_if_t<detail::is_array_convertible_v<U, T>
                                  && Extent != dynamic_extent && E2 == dynamic_extent,
                              bool> = true>
    CELER_CONSTEXPR_FUNCTION explicit Span(Span<U, E2> const& other) noexcept(
        ndebug_or_dyn)
        : s_(other.data(), other.size())
    {
    }

    CELER_DEFAULT_COPY_MOVE(Span);
    ~Span() = default;

    //// ACCESS ////

    //!@{
    //! \name Iterators
    CELER_CONSTEXPR_FUNCTION iterator begin() const { return s_.data; }
    CELER_CONSTEXPR_FUNCTION iterator end() const { return s_.data + s_.size; }
    //!@}

    //!@{
    //! \name Element access
    CELER_CONSTEXPR_FUNCTION reference operator[](std::size_t i) const
    {
        return s_.data[i];
    }
    CELER_CONSTEXPR_FUNCTION reference front() const { return s_.data[0]; }
    CELER_CONSTEXPR_FUNCTION reference back() const
    {
        return s_.data[s_.size - 1];
    }
    CELER_CONSTEXPR_FUNCTION pointer data() const
    {
        return static_cast<pointer>(s_.data);
    }
    //!@}

    //!@{
    //! \name Observers
    CELER_CONSTEXPR_FUNCTION bool empty() const { return s_.size == 0; }
    CELER_CONSTEXPR_FUNCTION std::size_t size() const { return s_.size; }
    CELER_CONSTEXPR_FUNCTION std::size_t size_bytes() const
    {
        return sizeof(element_type) * s_.size;
    }
    //!@}

    //!@{
    //! \name Subviews
    template<std::size_t Count>
    CELER_CONSTEXPR_FUNCTION Span<T, Count> first() const noexcept(ndebug)
    {
        CELER_EXPECT(Count == 0 || Count <= this->size());
        return {this->data(), Count};
    }
    CELER_CONSTEXPR_FUNCTION
    Span<T, dynamic_extent> first(std::size_t count) const noexcept(ndebug)
    {
        CELER_EXPECT(count <= this->size());
        return {this->data(), count};
    }

    template<std::size_t Offset, std::size_t Count = dynamic_extent>
    CELER_CONSTEXPR_FUNCTION
        Span<T, detail::subspan_extent(Extent, Offset, Count)>
        subspan() const noexcept(ndebug)
    {
        CELER_EXPECT((Count == dynamic_extent) || (Offset == 0 && Count == 0)
                     || (Offset + Count <= this->size()));
        return {this->data() + Offset,
                detail::subspan_size(this->size(), Offset, Count)};
    }
    CELER_CONSTEXPR_FUNCTION
    Span<T, dynamic_extent>
    subspan(std::size_t offset, std::size_t count = dynamic_extent) const
        noexcept(ndebug)
    {
        CELER_EXPECT(offset + count <= this->size());
        return {this->data() + offset,
                detail::subspan_size(this->size(), offset, count)};
    }

    template<std::size_t Count>
    CELER_CONSTEXPR_FUNCTION Span<T, Count> last() const noexcept(ndebug)
    {
        CELER_EXPECT(Count == 0 || Count <= this->size());
        return {this->data() + this->size() - Count, Count};
    }
    CELER_CONSTEXPR_FUNCTION
    Span<T, dynamic_extent> last(std::size_t count) const noexcept(ndebug)
    {
        CELER_EXPECT(count <= this->size());
        return {this->data() + this->size() - count, count};
    }
    //!@}

  private:
    //// DATA ////
    detail::SpanImpl<T, Extent> s_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

// Deduction guide for pointer and size
template<class T>
CELER_FUNCTION Span(T*, std::size_t) -> Span<T>;

// Deduction guide for two iterators
template<class Iter>
CELER_FUNCTION Span(Iter, Iter)
    -> Span<typename std::iterator_traits<Iter>::value_type>;

// Deduction guide for C array
template<class T, std::size_t N>
CELER_FUNCTION Span(T (&)[N]) -> Span<T, N>;

// Deduction guide for mutable Array
template<class T, std::size_t N>
CELER_FUNCTION Span(Array<T, N>&) -> Span<T, N>;

// Deduction guide for const Array
template<class T, std::size_t N>
CELER_FUNCTION Span(Array<T, N> const&) -> Span<T const, N>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Get a mutable fixed-size view to an array
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION Span<T, N> make_span(Array<T, N>& x)
{
    return {x.data(), N};
}

//---------------------------------------------------------------------------//
//! Get a constant fixed-size view to an array
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION Span<T const, N> make_span(Array<T, N> const& x)
{
    return {x.data(), N};
}

//---------------------------------------------------------------------------//
//! Get a mutable fixed-size view to a C array
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION Span<T, N> make_span(T (&arr)[N])
{
    return {arr, N};
}

//---------------------------------------------------------------------------//
//! Get a mutable view to a generic container
template<class T>
CELER_CONSTEXPR_FUNCTION Span<typename T::value_type> make_span(T& cont)
{
    return {cont.data(), cont.size()};
}

//---------------------------------------------------------------------------//
//! Get a const view to a generic container
template<class T>
CELER_CONSTEXPR_FUNCTION Span<typename T::value_type const>
make_span(T const& cont)
{
    return {cont.data(), cont.size()};
}

#if !CELER_DEVICE_COMPILE
//---------------------------------------------------------------------------//
/*!
 * Write the elements of array \a a to stream \a os.
 */
template<class T, std::size_t N>
CELER_FORCEINLINE std::ostream&
operator<<(std::ostream& os, Span<T, N> const& s)
{
    os << StreamableContainer{s.data(), s.size()};
    return os;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
