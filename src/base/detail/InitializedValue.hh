//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitializedValue.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Clear the value of the object on initialization and moving.
 */
template<class T>
class InitializedValue
{
  public:
    //! Construct with default value
    InitializedValue(T value = {}) : value_(value) {}

    //@{
    //! Default copy assign and construct
    InitializedValue(const InitializedValue&) noexcept = default;
    InitializedValue& operator=(const InitializedValue&) noexcept = default;
    //@}

    //! Clear other value on move construct
    InitializedValue(InitializedValue&& other) noexcept : value_(other.value_)
    {
        other.value_ = {};
    }

    //! Clear other value on move sassign
    InitializedValue& operator=(InitializedValue&& other) noexcept
    {
        value_       = other.value_;
        other.value_ = {};
        return *this;
    }

    //! Implicit assign from type
    InitializedValue operator=(const T& value) { value_ = value; }

    //! Implicit conversion to stored type
    operator T() const { return value_; }

  private:
    T value_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
