//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/io/StreamableContainer.hh"

#include "Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write the elements of array \a a to stream \a os.
 */
template<class T, std::size_t N>
CELER_FORCEINLINE std::ostream&
operator<<(std::ostream& os, Span<T, N> const& s)
{
    os << StreamableContainer{s.data(), s.size()};
    return os;
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
