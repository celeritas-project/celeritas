//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/SpanIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iomanip>
#include <ostream>

#include "Span.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write the elements of span \a s to stream \a os.
 */
template<class T, std::size_t E>
std::ostream& operator<<(std::ostream& os, Span<T, E> const& s)
{
    std::streamsize size = s.size();
    std::streamsize width = os.width();
    std::streamsize remainder = 0;

    os.width(0);
    os << '{';
    if (width > 2 + (size - 1))
    {
        // Subtract width for spaces and braces
        width -= 2 + (size - 1);
        // Individual width is 1/N of that, rounded down, keep remainder
        // separate
        remainder = width % size;
        width = width / size;
    }
    else
    {
        width = 0;
    }

    // First element gets the remainder
    os.width(width + remainder);
    if (!s.empty())
    {
        os << s[0];
    }

    for (std::streamsize i = 1; i < size; ++i)
    {
        os << ',';
        os.width(width);
        os << s[i];
    }
    os << '}';

    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
