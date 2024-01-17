//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/TypeDemangler.cc
//---------------------------------------------------------------------------//
#include "TypeDemangler.hh"

#ifdef __GNUG__
#    include <cstdlib>
#    include <cxxabi.h>
#endif  // __GNUG__

namespace celeritas
{
//---------------------------------------------------------------------------//
std::string demangled_typeid_name(char const* typeid_name)
{
#ifdef __GNUG__
    int status = -1;
    // Return a null-terminated string allocated with malloc
    char* demangled
        = abi::__cxa_demangle(typeid_name, nullptr, nullptr, &status);

    // Copy the C string to a STL string if successful, or the mangled name if
    // not
    std::string result(status == 0 ? demangled : typeid_name);

    // Free the returned memory
    std::free(demangled);
#else  // __GNUG__
    std::string result(typeid_name);
#endif  // __GNUG__

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
