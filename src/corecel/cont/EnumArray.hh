//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/EnumArray.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

namespace celeritas
{
#define CFIF_ CELER_FORCEINLINE_FUNCTION

//---------------------------------------------------------------------------//
/*!
 * Thin wrapper for an array of enums for accessing by enum instead of int.
 *
 * The enum *must* be a zero-indexed contiguous enumeration with a \c size_
 * enumeration as its last value.
 */
template<class E, class T>
struct EnumArray
{
    static_assert(std::is_enum<E>::value, "Template parameter must be an enum");

    //!@{
    //! Type aliases
    using key_type        = E;
    using value_type      = T;
    using size_type       = std::underlying_type_t<E>;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using iterator        = pointer;
    using const_iterator  = const_pointer;
    //!@}

    enum
    {
        N = static_cast<size_type>(key_type::size_)
    };

    //// DATA ////

    T data_[N]; //!< Storage

    //! Get an element
    CFIF_ reference operator[](const key_type& k)
    {
        return data_[static_cast<size_type>(k)];
    }

    //! Get an element (const)
    CFIF_ const_reference operator[](const key_type& k) const
    {
        return data_[static_cast<size_type>(k)];
    }

    //!@{
    //! Element access
    CFIF_ const_reference front() const { return data_[0]; }
    CFIF_ reference       front() { return data_[0]; }
    CFIF_ const_reference back() const { return data_[N - 1]; }
    CFIF_ reference       back() { return data_[N - 1]; }
    CFIF_ const_pointer   data() const { return data_; }
    CFIF_ pointer         data() { return data_; }
    //!@}

    //!@{
    //! Iterator acces
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
};

#undef CFIF_

//---------------------------------------------------------------------------//
} // namespace celeritas
