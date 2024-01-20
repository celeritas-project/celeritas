//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/ArrayIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <sstream>
#include <string>

#include "Array.hh"
#include "SpanIO.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write the elements of array \a a to stream \a os.
 */
template<class T, size_type N>
std::ostream& operator<<(std::ostream& os, Array<T, N> const& a)
{
    os << make_span(a);
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Convert an array to a string representation for debugging.
 */
template<class T, size_type N>
std::string to_string(Array<T, N> const& a)
{
    std::ostringstream os;
    os << a;
    return os.str();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
