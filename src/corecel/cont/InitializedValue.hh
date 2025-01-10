//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/InitializedValue.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>
#include <utility>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T>
struct DefaultFinalize
{
    void operator()(T&) const noexcept {}
};
//---------------------------------------------------------------------------//
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Clear the value of the object on initialization and moving.
 *
 * This helper class is used to simplify the "rule of 5" for classes that have
 * to treat one member data specially but can use default assign/construct for
 * the other elements. The default behavior is just to default-initialize when
 * assigning and clearing the RHS when moving; this is useful for handling
 * managed memory. The *finalizer* is useful when the type has a
 * destructor-type method that has to be called before clearing it.
 */
template<class T, class Finalizer = detail::DefaultFinalize<T>>
class InitializedValue
{
  private:
    static constexpr bool ne_finalize_
        = noexcept(std::declval<Finalizer>()(std::declval<T&>()));

  public:
    //!@{
    //! \name Constructors

    //! Construct implicitly with default value
    InitializedValue() = default;
    //! Implicit construct from lvalue
    InitializedValue(T const& value) : value_(value) {}
    //! Implicit construct from lvalue and finalizer
    InitializedValue(T const& value, Finalizer fin)
        : value_(value), fin_(std::move(fin))
    {
    }
    //! Implicit construct from rvalue
    InitializedValue(T&& value) : value_(std::move(value)) {}
    //! Implicit construct from value type and finalizer
    InitializedValue(T&& value, Finalizer fin)
        : value_(std::move(value)), fin_(std::move(fin))
    {
    }

    //! Default copy constructor
    InitializedValue(InitializedValue const&) noexcept(
        std::is_nothrow_copy_constructible_v<T>)
        = default;

    //! Clear other value on move construct
    InitializedValue(InitializedValue&& other) noexcept(
        std::is_nothrow_move_constructible_v<T>)
        : value_(std::exchange(other.value_, {}))
    {
    }

    ~InitializedValue() = default;

    //!@}
    //!@{
    //! \name Assignment

    //! Finalize our value when assigning
    InitializedValue& operator=(InitializedValue const& other) noexcept(
        ne_finalize_ && std::is_nothrow_copy_assignable_v<T>)
    {
        fin_(value_);
        value_ = other.value_;
        fin_ = other.fin_;
        return *this;
    }

    //! Clear other value on move assign
    InitializedValue& operator=(InitializedValue&& other) noexcept(
        ne_finalize_ && std::is_nothrow_move_assignable_v<T>)
    {
        fin_(value_);
        value_ = std::exchange(other.value_, {});
        fin_ = std::exchange(other.fin_, {});
        return *this;
    }

    //! Implicit assign from type
    InitializedValue&
    operator=(T const& value) noexcept(ne_finalize_
                                       && std::is_nothrow_copy_assignable_v<T>)
    {
        fin_(value_);
        value_ = value;
        return *this;
    }

    //! Swap with another value
    void swap(InitializedValue& other) noexcept
    {
        using std::swap;
        swap(other.value_, value_);
        swap(other.fin_, fin_);
    }

    //!@}
    //!@{
    //! \name Conversion

    //! Implicit reference to stored value
    operator T const&() const noexcept { return value_; }
    operator T&() noexcept { return value_; }

    //! Explicit reference to stored value
    T const& value() const& { return value_; }
    T& value() & { return value_; }
    T&& value() && { return value_; }

    //!@}

    //!@{
    //! Access finalizer
    Finalizer const& finalizer() const { return fin_; }
    void finalizer(Finalizer fin) { fin_ = std::move(fin); }
    //!@}

  private:
    T value_{};
    Finalizer fin_{};  // TODO: if C++20, [[no_unique_address]]
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
