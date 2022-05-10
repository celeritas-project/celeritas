//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/Joined.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iosfwd>
#include <iterator>
#include <sstream>
#include <string>

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
    Conjunction   conjunction;
    StreamOp      op;
};

//---------------------------------------------------------------------------//
template<class I, class C, class S>
std::ostream& operator<<(std::ostream& os, const Joined<I, C, S>& j)
{
    auto iter = j.first;
    auto op   = j.op;

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

//---------------------------------------------------------------------------//
template<class I, class C, class S>
std::string to_string(const Joined<I, C, S>& j)
{
    std::ostringstream os;
    os << j;
    return os.str();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
