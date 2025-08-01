//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
//---------------------------------------------------------------------------//
/*!
 * Thin wrapper for an array of enums for accessing by enum instead of int.
 *
 * The enum \em must be a zero-indexed contiguous enumeration with a \c size_
 * enumeration as its last value.
 *
 * \todo The template parameters are reversed!!!
 */
template<class E, class T>
class EnumArray
{
    static_assert(std::is_enum<E>::value, "Template parameter must be an enum");

  public:
    //!@{
    //! \name Type aliases
    using key_type = E;
    using value_type = T;
    using size_type = std::underlying_type_t<E>;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using reference = value_type&;
    using const_reference = value_type const&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    //!@}

    enum
    {
        N = static_cast<size_type>(E::size_)
    };

    using CArrayConstRef = T const (&)[N];

  public:
    //! Default construction initializes to zero
    CELER_CEF EnumArray() : data_{T{}} {}

    //! Construct from an array for aggregate initialization of daughters
    CELER_CEF EnumArray(CArrayConstRef values)
    {
        for (size_type i = 0; i < N; ++i)
        {
            data_[i] = values[i];
        }
    }

    //! Construct with C-style aggregate initialization
    EnumArray(T first) : data_{{first}} {}

    //! Construct with the array's data
    template<class... Us>
    CELER_CEF EnumArray(T first, Us... rest)
        : data_{first, static_cast<T>(rest)...}
    {
        // Protect against leaving off an entry, e.g. EnumArray<Foo, int>(1)
        static_assert(sizeof...(rest) + 1 == N,
                      "All array entries must be explicitly specified");
    }

    //! Get an element
    CELER_CEF reference operator[](key_type const& k)
    {
        return data_[static_cast<size_type>(k)];
    }

    //! Get an element (const)
    CELER_CEF const_reference operator[](key_type const& k) const
    {
        return data_[static_cast<size_type>(k)];
    }

    //!@{
    //! Element access
    CELER_CEF const_reference front() const { return data_[0]; }
    CELER_CEF reference front() { return data_[0]; }
    CELER_CEF const_reference back() const { return data_[N - 1]; }
    CELER_CEF reference back() { return data_[N - 1]; }
    CELER_CEF const_pointer data() const { return data_; }
    CELER_CEF pointer data() { return data_; }
    //!@}

    //!@{
    //! Iterator access
    CELER_CEF iterator begin() { return data_; }
    CELER_CEF iterator end() { return data_ + N; }
    CELER_CEF const_iterator begin() const { return data_; }
    CELER_CEF const_iterator end() const { return data_ + N; }
    CELER_CEF const_iterator cbegin() const { return data_; }
    CELER_CEF const_iterator cend() const { return data_ + N; }
    //!@}

    //!@{
    //! Capacity
    CELER_CEF bool empty() const { return N == 0; }
    CELER_CEF size_type size() const { return N; }
    //!@}

  private:
    T data_[N];  //!< Storage
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
