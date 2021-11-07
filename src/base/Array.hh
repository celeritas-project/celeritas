//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Array.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Macros.hh"
#include "Types.hh"

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
 * the range the size capacity.
 */
template<class T, ::celeritas::size_type N>
struct Array
{
    //!@{
    //! Type aliases
    using value_type      = T;
    using size_type       = ::celeritas::size_type;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using iterator        = pointer;
    using const_iterator  = const_pointer;
    //!@}

    //// DATA ////

    T data_[N]; //!< Storage

    //// ACCESSORS ////

    //!@{
    //! Element access
    CFIF_ const_reference operator[](size_type i) const { return data_[i]; }
    CFIF_ reference       operator[](size_type i) { return data_[i]; }
    CFIF_ const_reference front() const { return data_[0]; }
    CFIF_ reference       front() { return data_[0]; }
    CFIF_ const_reference back() const { return data_[N - 1]; }
    CFIF_ reference       back() { return data_[N - 1]; }
    CFIF_ const_pointer   data() const { return data_; }
    CFIF_ pointer         data() { return data_; }
    //!@}

    //!@{
    //! Iterators
    CFIF_ iterator       begin() { return data_; }
    CFIF_ iterator       end() { return data_ + N; }
    CFIF_ const_iterator begin() const { return data_; }
    CFIF_ const_iterator end() const { return data_ + N; }
    CFIF_ const_iterator cbegin() const { return data_; }
    CFIF_ const_iterator cend() const { return data_ + N; }
    //!@}

    //!@{
    //! Capacity
    CELER_CONSTEXPR_FUNCTION bool      empty() const { return N == 0; }
    CELER_CONSTEXPR_FUNCTION size_type size() const { return N; }
    //!@}

    //!@{
    //! Operations
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
inline CELER_FUNCTION bool
operator==(const Array<T, N>& lhs, const Array<T, N>& rhs)
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
CFIF_ bool operator!=(const Array<T, N>& lhs, const Array<T, N>& rhs)
{
    return !(lhs == rhs);
}

#undef CFIF_

//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//

//! Fixed-size array for R3 calculations
using Real3 = Array<real_type, 3>;

//---------------------------------------------------------------------------//
} // namespace celeritas
