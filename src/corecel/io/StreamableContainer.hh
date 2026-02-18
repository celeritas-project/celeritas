//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamableContainer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iomanip>
#include <ostream>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class to print a contiguous range of data.
 *
 * Since this class is used by both \c Array and \c Span, it is templated
 * solely on the item type, and directly uses a non-owning pointer with a size.
 * \code
   std::cout << StreamableContainer{s.data(), s.size()} << std::endl;
   \endcode
 */
template<class T>
struct StreamableContainer
{
    T const* data{};
    std::size_t size{};
};

//---------------------------------------------------------------------------//
// Template deduction guides
//---------------------------------------------------------------------------//

template<class T>
StreamableContainer(T const*, std::size_t) -> StreamableContainer<T>;

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Write a container to a stream.
 */
template<class T>
inline std::ostream& operator<<(std::ostream& os, StreamableContainer<T> sc)
{
    std::streamsize size = static_cast<std::streamsize>(sc.size);
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
    if (size > 0)
    {
        os << sc.data[0];
    }

    for (std::streamsize i = 1; i < size; ++i)
    {
        os << ',';
        os.width(width);
        os << sc.data[i];
    }
    os << '}';

    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
