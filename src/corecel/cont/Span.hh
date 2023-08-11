//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Span.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <type_traits>

#include "Array.hh"
#include "detail/SpanImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Sentinel value for span of dynamic type
constexpr std::size_t dynamic_extent = detail::dynamic_extent;

//---------------------------------------------------------------------------//
/*!
 * Modified backport of C++20 span.
 * \tparam T value type
 * \tparam Extent fixed size; defaults to dynamic.
 *
 * Like the \ref celeritas::Array , this class isn't 100% compatible
 * with the \c std::span class (partly of course because language features
 * are missing from C++14). The hope is that it will be complete and correct
 * for the use cases needed by Celeritas (and, as a bonus, it will be
 * device-compatible).
 *
 * Notably, only a subset of the functions (those having to do with size) are
 * \c constexpr. This is to allow debug assertions.
 */
template<class T, std::size_t Extent = dynamic_extent>
class Span
{
  public:
    //!@{
    //! \name Type aliases
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    //!@}

    //! Size (may be dynamic)
    static constexpr std::size_t extent = Extent;

  public:
    //// CONSTRUCTION ////

    //! Construct with default null pointer and size zero
    constexpr Span() = default;

    //! Construct from data and size
    CELER_FORCEINLINE_FUNCTION Span(pointer d, size_type s) : s_(d, s) {}

    //! Construct from two contiguous random-access iterators
    template<class Iter>
    CELER_FORCEINLINE_FUNCTION Span(Iter first, Iter last)
        : s_(&(*first), static_cast<size_type>(last - first))
    {
    }

    //! Construct from a C array
    template<std::size_t N>
    CELER_CONSTEXPR_FUNCTION Span(T (&arr)[N]) : s_(arr, N)
    {
    }

    //! Construct from another span
    template<class U, std::size_t N>
    CELER_CONSTEXPR_FUNCTION Span(Span<U, N> const& other)
        : s_(other.data(), other.size())
    {
    }

    //! Copy constructor (same template parameters)
    Span(Span const&) noexcept = default;

    //! Assignment (same template parameters)
    Span& operator=(Span const&) noexcept = default;

    //// ACCESS ////

    //!@{
    //! \name Iterators
    CELER_CONSTEXPR_FUNCTION iterator begin() const { return s_.data; }
    CELER_CONSTEXPR_FUNCTION iterator end() const { return s_.data + s_.size; }
    //!@}

    //!@{
    //! \name Element access
    CELER_CONSTEXPR_FUNCTION reference operator[](size_type i) const
    {
        return s_.data[i];
    }
    CELER_CONSTEXPR_FUNCTION reference front() const { return s_.data[0]; }
    CELER_CONSTEXPR_FUNCTION reference back() const
    {
        return s_.data[s_.size - 1];
    }
    CELER_CONSTEXPR_FUNCTION pointer data() const { return s_.data; }
    //!@}

    //!@{
    //! \name Observers
    CELER_CONSTEXPR_FUNCTION bool empty() const { return s_.size == 0; }
    CELER_CONSTEXPR_FUNCTION size_type size() const { return s_.size; }
    CELER_CONSTEXPR_FUNCTION size_type size_bytes() const
    {
        return sizeof(T) * s_.size;
    }
    //!@}

    //!@{
    //! \name Subviews
    template<std::size_t Count>
    CELER_FUNCTION Span<T, Count> first() const
    {
        CELER_EXPECT(Count == 0 || Count <= this->size());
        return {s_.data, Count};
    }
    CELER_FUNCTION
    Span<T, dynamic_extent> first(std::size_t count) const
    {
        CELER_EXPECT(count <= this->size());
        return {s_.data, count};
    }

    template<std::size_t Offset, std::size_t Count = dynamic_extent>
    CELER_FUNCTION Span<T, detail::subspan_extent(Extent, Offset, Count)>
    subspan() const
    {
        CELER_EXPECT((Count == dynamic_extent) || (Offset == 0 && Count == 0)
                     || (Offset + Count <= this->size()));
        return {s_.data + Offset,
                detail::subspan_size(this->size(), Offset, Count)};
    }
    CELER_FUNCTION
    Span<T, dynamic_extent>
    subspan(std::size_t offset, std::size_t count = dynamic_extent) const
    {
        CELER_EXPECT(offset + count <= this->size());
        return {s_.data + offset,
                detail::subspan_size(this->size(), offset, count)};
    }

    template<std::size_t Count>
    CELER_FUNCTION Span<T, Count> last() const
    {
        CELER_EXPECT(Count == 0 || Count <= this->size());
        return {this->data() + this->size() - Count, Count};
    }
    CELER_FUNCTION
    Span<T, dynamic_extent> last(std::size_t count) const
    {
        CELER_EXPECT(count <= this->size());
        return {this->data() + this->size() - count, count};
    }
    //!@}

  private:
    //! Storage
    detail::SpanImpl<T, Extent> s_;
};

template<class T, std::size_t N>
constexpr std::size_t Span<T, N>::extent;

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
CELER_CONSTEXPR_FUNCTION Span<const T, N> make_span(Array<T, N> const& x)
{
    return {x.data(), N};
}

//---------------------------------------------------------------------------//
//! Get a mutable fixed-size view to a C array
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION Span<T, N> make_span(T (&arr)[N])
{
    return {arr};
}

//---------------------------------------------------------------------------//
//! Get a mutable view to a generic container
template<class T>
CELER_FUNCTION Span<typename T::value_type> make_span(T& cont)
{
    return {cont.data(), cont.size()};
}

//---------------------------------------------------------------------------//
//! Get a const view to a generic container
template<class T>
CELER_FUNCTION Span<const typename T::value_type> make_span(T const& cont)
{
    return {cont.data(), cont.size()};
}

//---------------------------------------------------------------------------//
//! Construct an array from a fixed-size span
template<class T, std::size_t N>
CELER_CONSTEXPR_FUNCTION Array<std::remove_cv_t<T>, N>
make_array(Span<T, N> const& s)
{
    Array<std::remove_cv_t<T>, N> result;
    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = s[i];
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
