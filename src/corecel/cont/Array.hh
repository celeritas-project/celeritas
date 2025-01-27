//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Array.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <utility>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fixed-size simple array for storage.
 *
 * The Array class is primarily used for point coordinates (e.g., \c Real3) but
 * is also used for other fixed-size data structures.
 *
 * This isn't fully standards-compliant with std::array: there's no support for
 * N=0 for example. Additionally it uses the native celeritas \c size_type,
 * even though this has *no* effect on generated code for values of N inside
 * the range of \c size_type.
 *
 * \note For supplementary functionality, include:
 * - \c corecel/math/ArrayUtils.hh for real-number vector/matrix applications
 * - \c corecel/math/ArrayOperators.hh for mathematical operators
 * - \c ArrayIO.hh for streaming and string conversion
 * - \c ArrayIO.json.hh for JSON input and output
 */
template<class T, ::celeritas::size_type N>
struct Array
{
    //!@{
    //! \name Type aliases
    using value_type = T;
    using size_type = ::celeritas::size_type;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using reference = value_type&;
    using const_reference = value_type const&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    //!@}

    //// DATA ////

    T data_[N];  //!< Storage

    //// ACCESSORS ////

    //!@{
    //! \name Element access
    CELER_CONSTEXPR_FUNCTION const_reference operator[](size_type i) const
    {
        return data_[i];
    }
    CELER_CONSTEXPR_FUNCTION reference operator[](size_type i)
    {
        return data_[i];
    }
    CELER_CONSTEXPR_FUNCTION const_reference front() const { return data_[0]; }
    CELER_CONSTEXPR_FUNCTION reference front() { return data_[0]; }
    CELER_CONSTEXPR_FUNCTION const_reference back() const
    {
        return data_[N - 1];
    }
    CELER_CONSTEXPR_FUNCTION reference back() { return data_[N - 1]; }
    CELER_CONSTEXPR_FUNCTION const_pointer data() const { return data_; }
    CELER_CONSTEXPR_FUNCTION pointer data() { return data_; }

    //! Access for structured unpacking
    template<std::size_t I>
    CELER_CONSTEXPR_FUNCTION T& get()
    {
        static_assert(I < static_cast<std::size_t>(N));
        return data_[I];
    }

    //! Access for structured unpacking
    template<std::size_t I>
    CELER_CONSTEXPR_FUNCTION T const& get() const
    {
        static_assert(I < static_cast<std::size_t>(N));
        return data_[I];
    }
    //!@}

    //!@{
    //! \name Iterators
    CELER_CONSTEXPR_FUNCTION iterator begin() { return data_; }
    CELER_CONSTEXPR_FUNCTION iterator end() { return data_ + N; }
    CELER_CONSTEXPR_FUNCTION const_iterator begin() const { return data_; }
    CELER_CONSTEXPR_FUNCTION const_iterator end() const { return data_ + N; }
    CELER_CONSTEXPR_FUNCTION const_iterator cbegin() const { return data_; }
    CELER_CONSTEXPR_FUNCTION const_iterator cend() const { return data_ + N; }
    //!@}

    //!@{
    //! \name Capacity
    CELER_CONSTEXPR_FUNCTION bool empty() const { return N == 0; }
    static CELER_CONSTEXPR_FUNCTION size_type size() { return N; }
    //!@}

    //!@{
    //! \name  Operations

    //! Fill the array with a constant value
    CELER_CONSTEXPR_FUNCTION void fill(const_reference value)
    {
        for (size_type i = 0; i != N; ++i)
            data_[i] = value;
    }
    //!@}
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Test equality of two arrays.
 */
template<class T, size_type N>
CELER_CONSTEXPR_FUNCTION bool
operator==(Array<T, N> const& lhs, Array<T, N> const& rhs)
{
    for (size_type i = 0; i != N; ++i)
    {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * Test inequality of two arrays.
 */
template<class T, size_type N>
CELER_CONSTEXPR_FUNCTION bool
operator!=(Array<T, N> const& lhs, Array<T, N> const& rhs)
{
    return !(lhs == rhs);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
//! \cond
namespace std
{
//---------------------------------------------------------------------------//
//! Support structured binding: array size
template<class T, celeritas::size_type N>
struct tuple_size<celeritas::Array<T, N>>
{
    static constexpr std::size_t value = N;
};

//! Support structured binding: array element type
template<std::size_t I, class T, celeritas::size_type N>
struct tuple_element<I, celeritas::Array<T, N>>
{
    static_assert(I < std::tuple_size<celeritas::Array<T, N>>::value);
    using type = T;
};

//---------------------------------------------------------------------------//
}  // namespace std
//! \endcond
