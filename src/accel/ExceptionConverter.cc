//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/ExceptionConverter.cc
//---------------------------------------------------------------------------//
#include "ExceptionConverter.hh"

#include <sstream>
#include <G4Exception.hh>

#include "corecel/Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Capture the current exception and convert it to a G4Exception call.
 */
void ExceptionConverter::operator()(std::exception_ptr eptr)
{
    try
    {
        std::rethrow_exception(eptr);
    }
    catch (const RuntimeError& e)
    {
        // Translate a runtime error into a G4Exception call
        std::ostringstream where;
        if (e.details().file)
        {
            where << e.details().file;
        }
        if (e.details().line != 0)
        {
            where << ':' << e.details().line;
        }
        G4Exception(where.str().c_str(),
                    err_code_,
                    FatalException,
                    e.details().what.c_str());
    }
    catch (const DebugError& e)
    {
        // Translate a *debug* error
        std::ostringstream where;
        where << e.details().file << ':' << e.details().line;
        std::ostringstream what;
        what << to_cstring(e.details().which) << ": " << e.details().condition;
        G4Exception(
            where.str().c_str(), err_code_, FatalException, what.str().c_str());
    }
    // (Any other errors will be rethrown and abort the program.)
}

//---------------------------------------------------------------------------//
} // namespace celeritas
