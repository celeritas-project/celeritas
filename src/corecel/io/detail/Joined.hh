//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/Joined.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <string>

#include "corecel/io/StreamUtils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Default operator given an InputIterator: just stream the value.
 */
struct StreamValue
{
    template<class T>
    void operator()(std::ostream& os, T&& v)
    {
        os << std::forward<T>(v);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Convert a unary operator to a functor that writes to a stream.
 */
template<class UnaryOp>
struct UnaryToStream
{
    UnaryOp op;

    template<class T>
    void operator()(std::ostream& os, T&& v)
    {
        os << op(std::forward<T>(v));
    }
};

//---------------------------------------------------------------------------//
/*!
 * Implementation of joining a series of values.
 *
 * The advantage of this class is not having to create a temporary std::string
 * with the fully joined list.
 */
template<class InputIterator, class Conjunction, class StreamOp = StreamValue>
struct Joined
{
    InputIterator first;
    InputIterator last;
    Conjunction conjunction;
    StreamOp op;

    //! Write to a stream
    friend std::ostream& operator<<(std::ostream& os, Joined const& j)
    {
        auto iter = j.first;
        auto op = j.op;

        // First element is not preceded by a conjunction
        if (iter != j.last)
        {
            op(os, *iter++);
        }

        // Join the rest
        while (iter != j.last)
        {
            os << j.conjunction;
            op(os, *iter++);
        }

        return os;
    }

    //! Convert to a string
    friend std::string to_string(Joined const& j)
    {
        return stream_to_string(j);
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
