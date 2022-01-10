//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file EnumArray.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Types.hh"

namespace celeritas
{
#define CEL_FIF_ CELER_FORCEINLINE_FUNCTION

//---------------------------------------------------------------------------//
/*!
 * Thin wrapper for an array of enums for accessing by enum instead of int.
 */
template<class E, class T>
struct EnumArray
{
    enum
    {
        N = static_cast<int>(E::size_)
    };

    static_assert(std::is_enum<E>::value, "Template parameter must be an enum");

    //!@{
    //! Type aliases
    using key_type        = E;
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

    T data_[static_cast<int>(key_type::size_)]; //!< Storage

    //! Get an element
    CELER_FUNCTION reference operator[](const key_type& k)
    {
        CELER_EXPECT(k < key_type::size_);
        return data_[static_cast<int>(k)];
    }

    //! Get an element (const)
    CELER_FUNCTION const_reference operator[](const key_type& k) const
    {
        CELER_EXPECT(k < key_type::size_);
        return data_[static_cast<int>(k)];
    }

    //!@{
    //! Element access
    CEL_FIF_ const_reference front() const { return data_[0]; }
    CEL_FIF_ reference       front() { return data_[0]; }
    CEL_FIF_ const_reference back() const { return data_[N - 1]; }
    CEL_FIF_ reference       back() { return data_[N - 1]; }
    CEL_FIF_ const_pointer   data() const { return data_; }
    CEL_FIF_ pointer         data() { return data_; }
    //!@}

    //!@{
    //! Iterator acces
    CEL_FIF_ iterator       begin() { return data_; }
    CEL_FIF_ iterator       end() { return data_ + N; }
    CEL_FIF_ const_iterator begin() const { return data_; }
    CEL_FIF_ const_iterator end() const { return data_ + N; }
    CEL_FIF_ const_iterator cbegin() const { return data_; }
    CEL_FIF_ const_iterator cend() const { return data_ + N; }
    //!@}

    //!@{
    //! Capacity
    CELER_CONSTEXPR_FUNCTION bool      empty() const { return N == 0; }
    CELER_CONSTEXPR_FUNCTION size_type size() const { return N; }
    //!@}
};

#undef CEL_FIF_

//---------------------------------------------------------------------------//
} // namespace celeritas
