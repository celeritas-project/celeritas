//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Array.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

namespace celeritas
{
#define CFIF_ CELER_FORCEINLINE_FUNCTION

//---------------------------------------------------------------------------//
/*!
 * Fixed-size simple array for storage.
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
    CFIF_ const_reference operator[](size_type i) const { return data_[i]; }
    CFIF_ reference operator[](size_type i) { return data_[i]; }
    CFIF_ const_reference front() const { return data_[0]; }
    CFIF_ reference front() { return data_[0]; }
    CFIF_ const_reference back() const { return data_[N - 1]; }
    CFIF_ reference back() { return data_[N - 1]; }
    CFIF_ const_pointer data() const { return data_; }
    CFIF_ pointer data() { return data_; }
    //!@}

    //!@{
    //! \name Iterators
    CFIF_ iterator begin() { return data_; }
    CFIF_ iterator end() { return data_ + N; }
    CFIF_ const_iterator begin() const { return data_; }
    CFIF_ const_iterator end() const { return data_ + N; }
    CFIF_ const_iterator cbegin() const { return data_; }
    CFIF_ const_iterator cend() const { return data_ + N; }
    //!@}

    //!@{
    //! \name Capacity
    CELER_CONSTEXPR_FUNCTION bool empty() const { return N == 0; }
    static CELER_CONSTEXPR_FUNCTION size_type size() { return N; }
    //!@}

    //!@{
    //! \name  Operations
    //! Fill the array with a constant value
    CFIF_ void fill(const_reference value)
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

#undef CFIF_

//---------------------------------------------------------------------------//
}  // namespace celeritas
