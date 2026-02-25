//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//

// DEPRECATED: remove in 1.0
#warning "Deprecated: this file is no longer necessary to include"

#include "corecel/io/StreamToString.hh"

#include "Array.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Convert an array to a string representation for debugging.
 */
template<class T, size_type N>
std::string to_string(Array<T, N> const& a)
{
    return stream_to_string(a);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
