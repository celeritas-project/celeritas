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
 * Return a range over fixed beginning and end values.
 */
template<typename T>
CELER_FUNCTION detail::FiniteRange<T> range(T begin, T end)
{
    return {begin, end};
}

//---------------------------------------------------------------------------//
/*!
 * Return a range with the default start value (0 for numeric types)
 */
template<typename T>
CELER_FUNCTION detail::FiniteRange<T> range(T end)
{
    return {T(), end};
}

//---------------------------------------------------------------------------//
/*!
 * Count upward from zero.
 */
template<typename T>
CELER_FUNCTION detail::InfiniteRange<T> count()
{
    return {T()};
}

//---------------------------------------------------------------------------//
/*!
 * Count upward from a value.
 */
template<typename T>
CELER_FUNCTION detail::InfiniteRange<T> count(T begin)
{
    return {begin};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
