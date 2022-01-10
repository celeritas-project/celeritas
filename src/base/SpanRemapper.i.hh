//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SpanRemapper.i.hh
//---------------------------------------------------------------------------//

#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with source and destination ranges.
 */
template<class T, class U>
SpanRemapper<T, U>::SpanRemapper(src_type src_span, dst_type dst_span)
    : src_(src_span), dst_(dst_span)
{
    CELER_EXPECT(src_span.size() == dst_span.size());
}

//---------------------------------------------------------------------------//
/*!
 * Convert a subspan of the "source" to a corresponding subspan in "dst".
 */
template<class T, class U>
template<class V>
auto SpanRemapper<T, U>::operator()(Span<V> src_subspan) const -> dst_type
{
    CELER_EXPECT(src_subspan.empty()
                 || (src_subspan.begin() >= src_.begin()
                     && src_subspan.end() <= src_.end()));

    if (src_subspan.empty())
        return {};

    return dst_.subspan(src_subspan.data() - src_.data(), src_subspan.size());
}

//---------------------------------------------------------------------------//
} // namespace celeritas
