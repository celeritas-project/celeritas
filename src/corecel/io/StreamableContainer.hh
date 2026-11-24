//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamableContainer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>

namespace celeritas
{
namespace detail
{
struct StreamSizes
{
    std::streamsize size{};
    std::streamsize width{};
    std::streamsize remainder{};
};

inline StreamSizes calc_stream_sizes(std::streamsize width, std::size_t num_els)
{
    StreamSizes ss;
    ss.size = static_cast<std::streamsize>(num_els);
    ss.width = width;
    ss.remainder = 0;

    if (ss.width > 2 + (ss.size - 1))
    {
        // Subtract width for spaces and braces
        ss.width -= 2 + (ss.size - 1);
        // Individual width is 1/N of that, rounded down, keep remainder
        // separate
        ss.remainder = ss.width % ss.size;
        ss.width = ss.width / ss.size;
    }
    else
    {
        ss.width = 0;
    }
    return ss;
}
}  // namespace detail

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

    /*!
     * Write a container to a stream.
     */
    friend std::ostream& operator<<(std::ostream& os, StreamableContainer sc)
    {
        auto ss = detail::calc_stream_sizes(os.width(), sc.size);
        os.width(0);
        os << '{';
        // First element gets the remainder
        os.width(ss.width + ss.remainder);
        if (ss.size > 0)
        {
            os << sc.data[0];
        }

        // Print middle elements with width
        for (std::streamsize i = 1; i < ss.size; ++i)
        {
            os << ',';
            os.width(ss.width);
            os << sc.data[i];
        }

        // Final element has width 1
        os << '}';

        return os;
    }
};

// CTAD: needed for C++17
template<class T>
StreamableContainer(T const*, std::size_t) -> StreamableContainer<T>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
