//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Join.hh
//---------------------------------------------------------------------------//
#pragma once

#include "detail/Joined.hh"  // IWYU pragma: export

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Join items, similar to python's "str.join" method.
 *
 * This utility function will concatenate the values passed to it. The given
 * conjunction ONLY appears between values.
 *
 * \code
   cout << celeritas::join(foo.begin(), foo.end(), ", ") << endl
   \endcode
 *
 * The result is a thin class that is streamable. (It can explicitly be
 * converted to a string with the
 * \c to_string method). By doing this instead of returning a std::string,
 * large and dynamic containers can be e.g. saved to disk.
 */
template<class InputIterator, class Conjunction>
detail::Joined<InputIterator, Conjunction, detail::StreamValue>
join(InputIterator first, InputIterator last, Conjunction&& conjunction)
{
    return {first, last, std::forward<Conjunction>(conjunction), {}};
}

//---------------------------------------------------------------------------//
/*!
 * Join items transformed by a helper functor.
 *
 * This joins all given elements, inserting conjunction between them. The 'op'
 * operator must transform each element into a printable object. For example,
 * \code
      [](const std::pair<int, int>& item) { return item->first; }
 * \endcode
 * could be used to get the 'key' item of a map.
 */
template<class InputIterator, class Conjunction, class UnaryOperation>
detail::Joined<InputIterator, Conjunction, detail::UnaryToStream<UnaryOperation>>
join(InputIterator first,
     InputIterator last,
     Conjunction&& conjunction,
     UnaryOperation&& op)
{
    return {first,
            last,
            std::forward<Conjunction>(conjunction),
            {std::forward<UnaryOperation>(op)}};
}

//---------------------------------------------------------------------------//
/*!
 * Join using a functor that takes (ostream&, value).
 */
template<class InputIterator, class Conjunction, class StreamOp>
detail::Joined<InputIterator, Conjunction, StreamOp>
join_stream(InputIterator first,
            InputIterator last,
            Conjunction&& conjunction,
            StreamOp&& op)
{
    return {first,
            last,
            std::forward<Conjunction>(conjunction),
            std::forward<StreamOp>(op)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
