//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/ReprImpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cstddef>
#include <ostream>
#include <string_view>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
template<class T>
struct DefaultRepr;
}

template<class T>
detail::DefaultRepr<T> repr(T const& obj);

namespace detail
{
//---------------------------------------------------------------------------//
// STREAMABLE
//---------------------------------------------------------------------------//

//! Implementation of repr that forwards to ReprFormatter policy
template<class T>
struct DefaultRepr
{
    T const& obj;

    friend std::ostream& operator<<(std::ostream& os, DefaultRepr const& s)
    {
        return celeritas::ReprFormatter<T>{}(os, s.obj);
    }
};

//! Stream a string using repr-specific escaping and formatting.
struct ReprString
{
    char const* s;
    std::size_t size;

    static ReprString from_cstring(char const* s);
    friend std::ostream& operator<<(std::ostream& os, ReprString value);
};

//---------------------------------------------------------------------------//
// HELPER FUNCTION DECLARATIONS
//---------------------------------------------------------------------------//
void repr_char(std::ostream& os, char value);

void write_hex_char(std::ostream& os, unsigned char value);

void write_string_repr(std::ostream& os, std::string_view value);

//---------------------------------------------------------------------------//
}  // namespace detail

//! Default "repr" dispatches to the ReprFormatter policy class.
template<class T>
detail::DefaultRepr<T> repr(T const& obj)
{
    return {obj};
}

}  // namespace celeritas
