//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamableLazy.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <string>

#include "StreamUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Lazily evaluate a functor that streams content.
 *
 * \par Example:
 * \code
   std::cout << StreamableLazy{[]() {
        return do_something_expensive_and_printable();
    }} << std::endl;
   \endcode
 */
template<class F>
struct StreamableLazy
{
    F func;

    //! Write to stream
    inline friend std::ostream& operator<<(std::ostream& os,
                                           StreamableLazy const& lazy)
    {
        return (os << lazy.func());
    }

    //! Save as a string
    inline friend std::string to_string(StreamableLazy const& svar)
    {
        return stream_to_string(svar);
    }
};

//---------------------------------------------------------------------------//
// Deduction guide (C++17)
template<class F>
StreamableLazy(F&&) -> StreamableLazy<F>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
