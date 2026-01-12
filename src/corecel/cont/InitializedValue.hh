//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/InitializedValue.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <utility>

#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
struct DefaultFinalize
{
    CELER_FORCEINLINE void operator()(T&) const noexcept {}
};
//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Clear and finalize default values when moving and destroying.
 *
 * This helper class is used to simplify the "rule of 5" for classes that have
 * to treat one member data specially but can use default assign/construct for
 * the other elements. The default behavior is just to default-initialize when
 * assigning and clearing the RHS when moving; this is useful for handling
 * managed memory. The \em finalizer is called every time a \em non-default
 * value is lost by assignment or destruction.
 */
template<class T, class Finalizer = detail::DefaultFinalize<T>>
class InitializedValue
{
    static_assert(std::is_default_constructible_v<T>);
    static_assert(std::is_default_constructible_v<Finalizer>);
    static constexpr bool noexcept_finalize_
        = noexcept(std::declval<Finalizer>()(std::declval<T&>()));

  public:
    //!@{
    //! \name Constructors

    //! Construct implicitly with default value
    InitializedValue() = default;
    //! Implicit construct from lvalue
    InitializedValue(T const& value) : value_(value) {}
    //! Implicit construct from rvalue
    InitializedValue(T&& value) : value_(std::move(value)) {}

    //! Default copy constructor
    InitializedValue(InitializedValue const&) noexcept(
        std::is_nothrow_copy_constructible_v<T>)
        = default;

    // Move constructor
    InitializedValue(InitializedValue&& other) noexcept(
        std::is_nothrow_move_constructible_v<T>);

    // Destructor
    ~InitializedValue() noexcept(noexcept_finalize_);

    //!@}

    //!@{
    //! \name Assignment
    // Copy assignment
    InitializedValue& operator=(InitializedValue const& other) noexcept(
        noexcept_finalize_ && std::is_nothrow_copy_assignable_v<T>);
    // Move assignment
    InitializedValue& operator=(InitializedValue&& other) noexcept(
        noexcept_finalize_ && std::is_nothrow_move_assignable_v<T>);
    //!@}

    //!@{
    //! \name Conversion

    //! Implicit reference to stored value
    operator T const&() const noexcept { return value_; }
    operator T&() noexcept { return value_; }

    //! Explicit reference to stored value
    T const& value() const& { return value_; }
    T& value() & { return value_; }

    //!@}

  private:
    T value_{};
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//! Exchange with other value on move construct
template<class T, class Finalizer>
InitializedValue<T, Finalizer>::InitializedValue(
    InitializedValue&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
    : value_(std::exchange(other.value_, {}))
{
}

//---------------------------------------------------------------------------//
//! Call finalizer on destruct
template<class T, class Finalizer>
InitializedValue<T, Finalizer>::~InitializedValue() noexcept(noexcept_finalize_)
{
    if (value_ != T{})
    {
        Finalizer{}(value_);
    }
}

//---------------------------------------------------------------------------//
//! Finalize our value when assigning
template<class T, class Finalizer>
InitializedValue<T, Finalizer>&
InitializedValue<T, Finalizer>::operator=(InitializedValue const& other) noexcept(
    noexcept_finalize_ && std::is_nothrow_copy_assignable_v<T>)
{
    if (value_ != T{})
    {
        Finalizer{}(value_);
    }
    value_ = other.value_;
    return *this;
}

//---------------------------------------------------------------------------//
//! Clear other value on move assign
template<class T, class Finalizer>
InitializedValue<T, Finalizer>&
InitializedValue<T, Finalizer>::operator=(InitializedValue&& other) noexcept(
    noexcept_finalize_ && std::is_nothrow_move_assignable_v<T>)
{
    if (value_ != T{})
    {
        Finalizer{}(value_);
    }
    value_ = std::exchange(other.value_, T{});
    return *this;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
