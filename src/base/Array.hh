//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Array.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include "Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Fixed-size simple array for storage.
 *
 * This isn't fully standards-compliant with std::array: there's no support for
 * N=0 for example.
 */
template<typename T, std::size_t N>
struct Array
{
    //@{
    //! Type aliases
    using value_type      = T;
    using size_type       = std::size_t;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using iterator        = pointer;
    using const_iterator  = const_pointer;
    //@}

    /// DATA ///

    T data_[N];

    /// ACCESSORS ///

    //@{
    //! Element access
    CELER_FORCEINLINE_FUNCTION const_reference operator[](size_type i) const
    {
        return data_[i];
    }
    CELER_FORCEINLINE_FUNCTION reference operator[](size_type i)
    {
        return data_[i];
    }
    CELER_FORCEINLINE_FUNCTION const_reference front() const
    {
        return data_[0];
    }
    CELER_FORCEINLINE_FUNCTION reference       front() { return data_[0]; }
    CELER_FORCEINLINE_FUNCTION const_reference back() const
    {
        return data_[N - 1];
    }
    CELER_FORCEINLINE_FUNCTION reference     back() { return data_[N - 1]; }
    CELER_FORCEINLINE_FUNCTION const_pointer data() const { return data_; }
    CELER_FORCEINLINE_FUNCTION pointer       data() { return data_; }
    //@}

    //@{
    //! Iterators
    CELER_FORCEINLINE_FUNCTION iterator       begin() { return data_; }
    CELER_FORCEINLINE_FUNCTION iterator       end() { return data_ + N; }
    CELER_FORCEINLINE_FUNCTION const_iterator begin() const { return data_; }
    CELER_FORCEINLINE_FUNCTION const_iterator end() const { return data_ + N; }
    CELER_FORCEINLINE_FUNCTION const_iterator cbegin() const { return data_; }
    CELER_FORCEINLINE_FUNCTION const_iterator cend() const
    {
        return data_ + N;
    }
    //@}

    //@{
    //! Capacity
    CELER_CONSTEXPR_FUNCTION bool      empty() const { return N == 0; }
    CELER_CONSTEXPR_FUNCTION size_type size() const { return N; }
    //@}

    //@{
    //! Operations
    CELER_FORCEINLINE_FUNCTION void fill(const_reference value) const
    {
        for (size_type i = 0; i != N; ++i)
            data_[i] = value;
    }
    //@}
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Test equality of two arrays.
 */
template<typename T, std::size_t N>
inline CELER_FUNCTION bool
operator==(const Array<T, N>& lhs, const Array<T, N>& rhs)
{
    for (std::size_t i = 0; i != N; ++i)
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
template<typename T, std::size_t N>
CELER_FORCEINLINE_FUNCTION bool
operator!=(const Array<T, N>& lhs, const Array<T, N>& rhs)
{
    return !(lhs == rhs);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
