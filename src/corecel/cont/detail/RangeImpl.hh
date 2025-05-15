//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/detail/RangeImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/OpaqueId.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, class Enable = void>
struct RangeTypeTraits
{
    static_assert(std::is_integral_v<T>,
                  "Only integer types can be used in a range");
    using value_type = T;
    using counter_type = T;
    using difference_type = std::make_signed_t<counter_type>;

    template<class U>
    using common_type = typename std::common_type<T, U>::type;

    static CELER_CONSTEXPR_FUNCTION value_type zero() { return {}; }
    static CELER_CONSTEXPR_FUNCTION bool is_valid(value_type) { return true; }
    static CELER_CONSTEXPR_FUNCTION counter_type to_counter(value_type v)
    {
        return v;
    }
    static CELER_CONSTEXPR_FUNCTION value_type to_value(counter_type c)
    {
        return c;
    }
    static CELER_CONSTEXPR_FUNCTION value_type increment(value_type v,
                                                         difference_type i)
    {
        v += i;
        return v;
    }
    static CELER_CONSTEXPR_FUNCTION value_type decrement(value_type v,
                                                         difference_type i)
    {
        v -= i;
        return v;
    }
};

template<class T, class Enable = void>
struct EnumWithSize
{
    static CELER_CONSTEXPR_FUNCTION bool is_valid(T) { return true; }
};

template<class T>
struct EnumWithSize<T, typename std::enable_if<T::size_ >= 0>::type>
{
    static CELER_CONSTEXPR_FUNCTION bool is_valid(T value)
    {
        return value <= T::size_;
    }
};

//! Specialization for enums with a "size_" member
template<class T>
struct RangeTypeTraits<T, typename std::enable_if<std::is_enum<T>::value>::type>
{
    using value_type = T;
    using counter_type = typename std::underlying_type<T>::type;
    using difference_type = std::make_signed_t<counter_type>;
    template<class U>
    using common_type = value_type;

    static CELER_CONSTEXPR_FUNCTION value_type zero() { return {}; }
    static CELER_CONSTEXPR_FUNCTION bool is_valid(value_type v)
    {
        return EnumWithSize<T>::is_valid(v);
    }
    static CELER_CONSTEXPR_FUNCTION counter_type to_counter(value_type v)
    {
        return static_cast<counter_type>(v);
    }
    static CELER_CONSTEXPR_FUNCTION value_type to_value(counter_type c)
    {
        return static_cast<value_type>(c);
    }
    static CELER_CONSTEXPR_FUNCTION value_type increment(value_type v,
                                                         difference_type i)
    {
        auto temp = to_counter(v);
        temp += i;
        return to_value(temp);
    }
    static CELER_CONSTEXPR_FUNCTION value_type decrement(value_type v,
                                                         difference_type i)
    {
        auto temp = to_counter(v);
        temp -= i;
        return to_value(temp);
    }
};

//! Specialization for Opaque ID
template<class I, class T>
struct RangeTypeTraits<OpaqueId<I, T>, void>
{
    using value_type = OpaqueId<I, T>;
    using counter_type = T;
    using difference_type = std::make_signed_t<counter_type>;
    template<class U>
    using common_type = value_type;

    static CELER_CONSTEXPR_FUNCTION value_type zero() { return value_type(0); }
    static CELER_CONSTEXPR_FUNCTION bool is_valid(value_type v)
    {
        return static_cast<bool>(v);
    }
    static CELER_CONSTEXPR_FUNCTION counter_type to_counter(value_type v)
    {
        return v.unchecked_get();
    }
    static CELER_CONSTEXPR_FUNCTION value_type to_value(difference_type c)
    {
        return static_cast<value_type>(c);
    }
    static CELER_CONSTEXPR_FUNCTION value_type increment(value_type v,
                                                         difference_type i)
    {
        counter_type temp = to_counter(v);
        temp += i;
        return to_value(temp);
    }
    static CELER_CONSTEXPR_FUNCTION value_type decrement(value_type v,
                                                         difference_type i)
    {
        counter_type temp = to_counter(v);
        temp -= i;
        return to_value(temp);
    }
};

//---------------------------------------------------------------------------//
template<class T>
class range_iter
{
  public:
    //!@{
    //! \name Type aliases
    using TraitsT = RangeTypeTraits<T>;
    using value_type = T;
    using counter_type = typename TraitsT::counter_type;
    using iterator_category = std::input_iterator_tag;
    using difference_type = typename TraitsT::difference_type;
    using pointer = void;
    using reference = value_type;
    //!@}

  public:
    //// CONSTRUCTOR ////

    CELER_FORCEINLINE_FUNCTION range_iter(value_type value) : value_(value)
    {
        CELER_EXPECT(TraitsT::is_valid(value_));
    }

    CELER_CONSTEXPR_FUNCTION range_iter() : value_(TraitsT::zero()) {}

    //// ACCESSORS ////

    CELER_CONSTEXPR_FUNCTION value_type operator*() const { return value_; }
    CELER_CONSTEXPR_FUNCTION value_type const* operator->() const
    {
        return &value_;
    }

    CELER_CONSTEXPR_FUNCTION value_type operator[](counter_type inc) const
    {
        return TraitsT::increment(value_, inc);
    }

    //// ARITHMETIC ////

    CELER_CONSTEXPR_FUNCTION range_iter& operator++()
    {
        value_ = TraitsT::increment(value_, 1);
        return *this;
    }

    CELER_CONSTEXPR_FUNCTION range_iter operator++(int)
    {
        auto copy = *this;
        value_ = TraitsT::increment(value_, 1);
        return copy;
    }

    CELER_CONSTEXPR_FUNCTION range_iter operator+(difference_type inc) const
    {
        return {TraitsT::increment(value_, inc)};
    }

    CELER_CONSTEXPR_FUNCTION range_iter& operator--()
    {
        value_ = TraitsT::decrement(value_, 1);
        return *this;
    }

    CELER_CONSTEXPR_FUNCTION range_iter operator--(int)
    {
        auto copy = *this;
        value_ = TraitsT::decrement(value_, 1);
        return copy;
    }

    CELER_CONSTEXPR_FUNCTION range_iter operator-(difference_type inc) const
    {
        return {TraitsT::decrement(value_, inc)};
    }

    CELER_CONSTEXPR_FUNCTION bool operator==(range_iter const& other) const
    {
        return value_ == other.value_;
    }

    CELER_CONSTEXPR_FUNCTION bool operator!=(range_iter const& other) const
    {
        return !(*this == other);
    }

    // Access the underlying value
    CELER_CONSTEXPR_FUNCTION value_type value() const { return value_; }

  protected:
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    value_type value_;
};

// Subtract two range iterators
template<class T>
CELER_CONSTEXPR_FUNCTION auto operator-(range_iter<T> a, range_iter<T> b)
{
    using TraitsT = RangeTypeTraits<T>;
    using DT = typename TraitsT::difference_type;
    return static_cast<DT>(TraitsT::to_counter(a.value()))
           - static_cast<DT>(TraitsT::to_counter(b.value()));
}

//---------------------------------------------------------------------------//
template<class T>
class inf_range_iter : public range_iter<T>
{
    using Base = range_iter<T>;

  public:
    using TraitsT = typename Base::TraitsT;

    CELER_CONSTEXPR_FUNCTION inf_range_iter(T value = TraitsT::zero())
        : Base(value)
    {
    }

    CELER_CONSTEXPR_FUNCTION bool operator==(inf_range_iter const&) const
    {
        return false;
    }

    CELER_CONSTEXPR_FUNCTION bool operator!=(inf_range_iter const&) const
    {
        return true;
    }
};

//---------------------------------------------------------------------------//
template<class T>
class step_range_iter : public range_iter<T>
{
    using Base = range_iter<T>;

  public:
    using TraitsT = typename Base::TraitsT;
    using counter_type = typename TraitsT::counter_type;
    using difference_type = typename TraitsT::difference_type;

    CELER_CONSTEXPR_FUNCTION step_range_iter(T value, counter_type step)
        : Base(value), step_(step)
    {
    }

    CELER_CONSTEXPR_FUNCTION step_range_iter& operator++()
    {
        value_ = TraitsT::increment(value_, step_);
        return *this;
    }

    CELER_CONSTEXPR_FUNCTION step_range_iter operator++(int)
    {
        auto copy = *this;
        ++*this;
        return copy;
    }

    CELER_CONSTEXPR_FUNCTION step_range_iter operator+(difference_type inc)
    {
        return {TraitsT::increment(value_, inc * step_)};
    }

    CELER_FUNCTION bool operator==(step_range_iter const& other) const
    {
        if constexpr (std::is_signed_v<counter_type>)
        {
            return step_ >= 0 ? !(value_ < other.value_)
                              : value_ < other.value_;
        }
        else
        {
            return !(value_ < other.value_);
        }
    }

    CELER_FUNCTION bool operator!=(step_range_iter const& other) const
    {
        return !(*this == other);
    }

    // Access the underlying step
    CELER_FORCEINLINE_FUNCTION counter_type step() const { return step_; }

  private:
    using Base::value_;
    counter_type step_;
};

//---------------------------------------------------------------------------//
template<class T>
class inf_step_range_iter : public step_range_iter<T>
{
    using Base = step_range_iter<T>;

  public:
    using TraitsT = typename Base::TraitsT;
    using counter_type = typename TraitsT::counter_type;

    CELER_FUNCTION
    inf_step_range_iter(T current = TraitsT::zero(), counter_type step = {})
        : Base(current, step)
    {
    }

    CELER_FUNCTION bool operator==(inf_step_range_iter const&) const
    {
        return false;
    }

    CELER_FUNCTION bool operator!=(inf_step_range_iter const&) const
    {
        return true;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Proxy container for iterating over a finite range with a non-unit step
 */
template<class T>
class StepRange
{
  public:
    using IterT = step_range_iter<T>;
    using counter_type = typename RangeTypeTraits<T>::counter_type;

    CELER_FUNCTION StepRange(T begin, T end, counter_type step)
        : begin_(begin, step), end_(end, step)
    {
    }

    CELER_CONSTEXPR_FUNCTION IterT begin() const { return begin_; }
    CELER_CONSTEXPR_FUNCTION IterT end() const { return end_; }

  private:
    IterT begin_;
    IterT end_;
};

//---------------------------------------------------------------------------//
/*!
 * Proxy container for iterating over an infinite range with a non-unit step
 */
template<class T>
class InfStepRange
{
  public:
    using IterT = inf_step_range_iter<T>;
    using counter_type = typename RangeTypeTraits<T>::counter_type;

    //! Construct from start/stop
    CELER_FUNCTION InfStepRange(T begin, counter_type step)
        : begin_(begin, step)
    {
    }

    CELER_CONSTEXPR_FUNCTION IterT begin() const { return begin_; }
    CELER_CONSTEXPR_FUNCTION IterT end() const { return IterT(); }

  private:
    IterT begin_;
};

}  // namespace detail
}  // namespace celeritas
