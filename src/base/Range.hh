//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Range.hh
//---------------------------------------------------------------------------//
#pragma once

#include "detail/RangeImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \fn range
 * \tparam T Value type to iterate over
 * Get iterators over a range of values, or a semi-infinite range.
 *
 * \par Code Sample:
 * \code

    for (auto i : Range(1, 5))
        cout << i << "\n";

    // Range of [0, 10)
    for (auto u : Range(10u))
        cout << u << "\n";

    for (auto c : Range('a', 'd'))
        cout << c << "\n";

    for (auto i : Count(100).step(-3))
        if (i < 90) break;
        else        cout << i << "\n";

 * \endcode
 */

//---------------------------------------------------------------------------//
/*!
 * Proxy container for iterating over a range of integral values.
 *
 * Here, T can be any of:
 * - an integer,
 * - an enum that has a "size_" member, or
 * - an OpaqueId.
 *
 * It is OK to dereference the end iterator! The result should just be the
 * off-the-end value for the range, e.g. `FooEnum::size_` or `bar.size()`.
 */
template<class T>
class Range
{
    using TraitsT = detail::RangeTypeTraits<T>;

  public:
    //!@{
    //! Type aliases
    using const_iterator = detail::range_iter<T>;
    using size_type      = typename TraitsT::counter_type;
    using value_type     = T;
    //@}

    template<class U>
    using step_type = typename TraitsT::template common_type<U>;

  public:
    //// CONSTRUCTORS ////

    //! Empty constructor for empty range
    CELER_FUNCTION Range() : begin_{}, end_{} {}

    //! Construct from stop
    CELER_FUNCTION Range(T end) : begin_{}, end_(end) {}

    //! Construct from start/stop
    CELER_FUNCTION Range(T begin, T end) : begin_(begin), end_(end) {}

    //// CONTAINER-LIKE ACCESS ////

    //!@{
    //! Iterators
    CELER_FORCEINLINE_FUNCTION const_iterator begin() const { return begin_; }
    CELER_FORCEINLINE_FUNCTION const_iterator cbegin() const { return begin_; }
    CELER_FORCEINLINE_FUNCTION const_iterator end() const { return end_; }
    CELER_FORCEINLINE_FUNCTION const_iterator cend() const { return end_; }
    //!@}

    //! Array-like access
    CELER_FORCEINLINE_FUNCTION value_type operator[](size_type i) const
    {
        return *(begin_ + i);
    }

    //! Number of elements
    CELER_FORCEINLINE_FUNCTION size_type size() const
    {
        return TraitsT::to_counter(*end_) - TraitsT::to_counter(*begin_);
    }

    //! Whether the range has no elements
    CELER_FORCEINLINE_FUNCTION bool empty() const { return begin_ == end_; }

    //! First item in the range
    CELER_FORCEINLINE_FUNCTION value_type front() const { return *begin_; }

    //! Last item in the range
    CELER_FORCEINLINE_FUNCTION value_type back() const
    {
        return (*this)[this->size() - 1];
    }

    //// STRIDED ACCESS ////

    //! Return a stepped range using a different integer type
    template<class U, std::enable_if_t<std::is_signed<U>::value, U> = 0>
    CELER_FUNCTION detail::StepRange<step_type<U>> step(U step)
    {
        if (step < 0)
        {
            return {TraitsT::increment(*end_, step), *begin_, step};
        }

        return {*begin_, *end_, step};
    }

    //! Return a stepped range using a different integer type
    template<class U, std::enable_if_t<std::is_unsigned<U>::value, U> = 0>
    CELER_FUNCTION detail::StepRange<step_type<U>> step(U step)
    {
        return {*begin_, *end_, step};
    }

  private:
    const_iterator begin_;
    const_iterator end_;
};

//---------------------------------------------------------------------------//
/*!
 * Proxy container for an unbounded range with a given start value.
 */
template<class T>
class Count
{
    using TraitsT = detail::RangeTypeTraits<T>;

  public:
    //!@{
    //! Type aliases
    using const_iterator = detail::inf_range_iter<T>;
    using size_type      = typename TraitsT::counter_type;
    using value_type     = T;
    //@}

    CELER_FUNCTION Count() : begin_{} {}
    CELER_FUNCTION Count(T begin) : begin_(begin) {}

    CELER_FUNCTION detail::InfStepRange<T> step(T step)
    {
        return {*begin_, step};
    }

    CELER_FUNCTION const_iterator begin() const { return begin_; }
    CELER_FUNCTION const_iterator end() const { return const_iterator(); }
    CELER_FUNCTION bool           empty() const { return false; }

  private:
    const_iterator begin_;
};

//---------------------------------------------------------------------------//
/*!
 * Return a range over fixed beginning and end values.
 */
template<class T>
CELER_FUNCTION Range<T> range(T begin, T end)
{
    return {begin, end};
}

//---------------------------------------------------------------------------//
/*!
 * Return a range with the default start value (0 for numeric types)
 */
template<class T>
CELER_FUNCTION Range<T> range(T end)
{
    return {end};
}

//---------------------------------------------------------------------------//
/*!
 * Count upward from zero.
 */
template<class T>
CELER_FUNCTION Count<T> count()
{
    return {};
}

//---------------------------------------------------------------------------//
/*!
 * Count upward from a value.
 */
template<class T>
CELER_FUNCTION Count<T> count(T begin)
{
    return {begin};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
