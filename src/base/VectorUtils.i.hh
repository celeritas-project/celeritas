//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VectorUtils.i.hh
//---------------------------------------------------------------------------//

#include <iterator>
#include "Assert.hh"

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
} // namespace celeritas
