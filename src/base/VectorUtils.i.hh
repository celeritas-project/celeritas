//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VectorUtils.i.hh
//---------------------------------------------------------------------------//

#include <iterator>
#include "Assert.hh"
#include "base/Range.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Copy the given extension to the end of a preallocated vector.
 */
template<class T, std::size_t N, class U>
Span<U> extend(Span<T, N> ext, std::vector<U>* base)
{
    CELER_EXPECT(base);
    CELER_EXPECT(base->size() + ext.size() <= base->capacity());

    auto start = base->size();
    base->insert(base->end(), ext.begin(), ext.end());
    return {base->data() + start, base->data() + base->size()};
}

//---------------------------------------------------------------------------//
/*!
 * Copy the given extension to the end of a preallocated vector.
 */
template<class T>
Span<T> extend(const std::vector<T>& ext, std::vector<T>* base)
{
    return extend(make_span(ext), base);
}

//---------------------------------------------------------------------------//
/*!
 * Move the given extension to the end of a vector.
 *
 * The vector does not have to be preallocated. The given data's elements will
 * be moved, and the extension vector will be erased on completion.
 */
template<class T>
Span<T> move_extend(std::vector<T>&& ext, std::vector<T>* base)
{
    CELER_EXPECT(base);

    auto start = base->size();
    base->insert(base->end(),
                 std::make_move_iterator(ext.begin()),
                 std::make_move_iterator(ext.end()));
    ext.clear();
    return {base->data() + start, base->data() + base->size()};
}

//---------------------------------------------------------------------------//
/*!
 * Return evenly spaced numbers over a specific interval
 */
template<class T>
std::vector<real_type> linspace(T start, T stop, size_type n)
{
    CELER_EXPECT(n > 1);
    std::vector<real_type> result(n);

    // Convert input values to real_type
    real_type start_c = start;
    real_type stop_c  = stop;

    // Build vector of evenly spaced numbers
    real_type delta = (stop_c - start_c) / (n - 1);
    for (auto i : range(n - 1))
    {
        result[i] = start_c + delta * i;
    }
    // Manually add last point to avoid any differences due to roundoff
    result[n - 1] = stop_c;
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
