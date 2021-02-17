//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RangeImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iterator>
#include <type_traits>
#include <utility>
#include "base/Assert.hh"
#include "base/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class T, class Enable = void>
struct RangeTypeTraits
{
    using value_type   = T;
    using counter_type = T;

    template<class U>
    using common_type = typename std::common_type<T, U>::type;

    static CELER_CONSTEXPR_FUNCTION bool is_valid(value_type) { return true; }
    static CELER_CONSTEXPR_FUNCTION counter_type to_counter(value_type v)
    {
        return v;
    }
    static CELER_CONSTEXPR_FUNCTION value_type to_value(counter_type c)
    {
        return c;
    }
    static CELER_FORCEINLINE_FUNCTION value_type increment(value_type   v,
                                                           counter_type i)
    {
        v += i;
        return v;
    }
};

//! Specialization for enums with a "size_" member
template<class T>
struct RangeTypeTraits<T, typename std::enable_if<std::is_enum<T>::value>::type>
{
    using value_type   = T;
    using counter_type = typename std::underlying_type<T>::type;
    template<class U>
    using common_type = value_type;

    static CELER_CONSTEXPR_FUNCTION bool is_valid(value_type v)
    {
        return v <= T::size_;
    }
    static CELER_CONSTEXPR_FUNCTION counter_type to_counter(value_type v)
    {
        return static_cast<counter_type>(v);
    }
    static CELER_CONSTEXPR_FUNCTION value_type to_value(counter_type c)
    {
        return static_cast<value_type>(c);
    }
    static CELER_FORCEINLINE_FUNCTION value_type increment(value_type   v,
                                                           counter_type i)
    {
        counter_type temp = to_counter(v);
        temp += i;
        return to_value(temp);
    }
};

//---------------------------------------------------------------------------//
template<class T>
class range_iter : public std::iterator<std::input_iterator_tag, T>
{
  public:
    //!@{
    //! Type aliases
    using TraitsT      = RangeTypeTraits<T>;
    using value_type   = typename TraitsT::value_type;
    using counter_type = typename TraitsT::counter_type;
    //!@}

  protected:
    value_type value_;

  public:
    //// CONSTRUCTOR ////

    CELER_FUNCTION range_iter(value_type value) : value_(value)
    {
        CELER_EXPECT(TraitsT::is_valid(value_));
    }

    //// ACCESSORS ////

    CELER_FORCEINLINE_FUNCTION value_type operator*() const { return value_; }
    CELER_FORCEINLINE_FUNCTION value_type const* operator->() const
    {
        return &value_;
    }

    //// ARITHMETIC ////

    CELER_FORCEINLINE_FUNCTION range_iter& operator++()
    {
        value_ = TraitsT::increment(value_, 1);
        return *this;
    }

    CELER_FORCEINLINE_FUNCTION range_iter operator++(int)
    {
        auto copy = *this;
        value_    = TraitsT::increment(value_, 1);
        return copy;
    }

    CELER_FORCEINLINE_FUNCTION bool operator==(range_iter const& other) const
    {
        return value_ == other.value_;
    }

    CELER_FORCEINLINE_FUNCTION bool operator!=(range_iter const& other) const
    {
        return !(*this == other);
    }
};

//---------------------------------------------------------------------------//
template<class T>
class inf_range_iter : public range_iter<T>
{
    using Base = range_iter<T>;

  public:
    CELER_FORCEINLINE_FUNCTION inf_range_iter(T value_ = T()) : Base(value_) {}

    CELER_FORCEINLINE_FUNCTION bool operator==(inf_range_iter const&) const
    {
        return false;
    }

    CELER_FORCEINLINE_FUNCTION bool operator!=(inf_range_iter const&) const
    {
        return true;
    }
};

//---------------------------------------------------------------------------//
template<class T>
class step_range_iter : public range_iter<T>
{
    using Base = range_iter<T>;
    using TraitsT      = typename Base::TraitsT;
    using counter_type = typename TraitsT::counter_type;

  public:
    CELER_FUNCTION step_range_iter(T value, counter_type step)
        : Base(value), step_(step)
    {
    }

    CELER_FUNCTION step_range_iter& operator++()
    {
        value_ = TraitsT::increment(value_, step_);
        return *this;
    }

    CELER_FUNCTION step_range_iter operator++(int)
    {
        auto copy = *this;
        ++*this;
        return copy;
    }

    template<class U = counter_type>
    CELER_FUNCTION typename std::enable_if_t<std::is_signed<U>::value, bool>
    operator==(step_range_iter const& other) const
    {
        return step_ >= 0 ? !(value_ < other.value_) : value_ < other.value_;
    }

    template<class U = counter_type>
    CELER_FUNCTION typename std::enable_if_t<std::is_unsigned<U>::value, bool>
    operator==(step_range_iter const& other) const
    {
        return !(value_ < other.value_);
    }

    CELER_FUNCTION bool operator!=(step_range_iter const& other) const
    {
        return !(*this == other);
    }

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
    CELER_FUNCTION inf_step_range_iter(T current = T(), T step = T())
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

    CELER_FORCEINLINE_FUNCTION IterT begin() const { return begin_; }
    CELER_FORCEINLINE_FUNCTION IterT end() const { return end_; }

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

    CELER_FORCEINLINE_FUNCTION IterT begin() const { return begin_; }
    CELER_FORCEINLINE_FUNCTION IterT end() const { return IterT(); }

  private:
    IterT begin_;
};

//---------------------------------------------------------------------------//
/*!
 * Proxy container for iterating over a range of integral values.
 */
template<class T>
class FiniteRange
{
  public:
    using IterT        = range_iter<T>;
    using TraitsT      = RangeTypeTraits<T>;
    using counter_type = typename TraitsT::counter_type;
    template<class U>
    using step_type = typename TraitsT::template common_type<U>;

    //! Empty constructor for empty range
    CELER_FUNCTION FiniteRange() : begin_(T()), end_(T()) {}

    //! Construct from start/stop
    CELER_FUNCTION FiniteRange(T begin, T end) : begin_(begin), end_(end) {}

    //! Return a stepped range using a different integer type
    template<class U, std::enable_if_t<std::is_signed<U>::value, U> = 0>
    CELER_FUNCTION StepRange<step_type<U>> step(U step)
    {
        if (step < 0)
        {
            using TraitsT = typename IterT::TraitsT;
            return {TraitsT::increment(*end_, step), *begin_, step};
        }

        return {*begin_, *end_, step};
    }

    //! Return a stepped range using a different integer type
    template<class U, std::enable_if_t<std::is_unsigned<U>::value, U> = 0>
    CELER_FUNCTION StepRange<step_type<U>> step(U step)
    {
        return {*begin_, *end_, step};
    }

    CELER_FUNCTION IterT        begin() const { return begin_; }
    CELER_FUNCTION IterT        end() const { return end_; }
    CELER_FUNCTION counter_type size() const
    {
        return TraitsT::to_counter(*end_) - TraitsT::to_counter(*begin_);
    }
    CELER_FUNCTION bool         empty() const { return end_ == begin_; }

  private:
    IterT begin_;
    IterT end_;
};

//---------------------------------------------------------------------------//
/*!
 * Proxy container for iterating over a range of integral values.
 */
template<class T>
class InfiniteRange
{
  public:
    using IterT = inf_range_iter<T>;

    CELER_FUNCTION InfiniteRange(T begin) : begin_(begin) {}

    CELER_FUNCTION InfStepRange<T> step(T step) { return {*begin_, step}; }

    CELER_FUNCTION IterT begin() const { return begin_; }
    CELER_FUNCTION IterT end() const { return IterT(); }
    CELER_FUNCTION bool  empty() const { return false; }

  private:
    IterT begin_;
};

} // namespace detail
} // namespace celeritas
