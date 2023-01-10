//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/detail/InitializedValue.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

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
    //! Construct implicitly with default value
    InitializedValue() = default;
    //! Implicit construct from value type
    InitializedValue(T value) : value_(std::move(value)) {}

    //!@{
    //! Default copy assign and construct
    InitializedValue(InitializedValue const&) noexcept = default;
    InitializedValue& operator=(InitializedValue const&) noexcept = default;
    //!@}

    //! Clear other value on move construct
    InitializedValue(InitializedValue&& other) noexcept
        : value_(std::exchange(other.value_, {}))
    {
    }

    //! Clear other value on move assign
    InitializedValue& operator=(InitializedValue&& other) noexcept
    {
        value_ = std::exchange(other.value_, {});
        return *this;
    }

    //! Implicit assign from type
    InitializedValue& operator=(T const& value)
    {
        value_ = value;
        return *this;
    }

    //! Implicit conversion to stored type
    operator T() const { return value_; }

    //! Swap with another value
    void swap(InitializedValue& other) noexcept
    {
        using std::swap;
        swap(other.value_, value_);
    }

  private:
    T value_{};
};

//---------------------------------------------------------------------------//
/*!
 * Swap two values.
 */
template<class T>
inline void swap(InitializedValue<T>& a, InitializedValue<T>& b) noexcept
{
    a.swap(b);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
