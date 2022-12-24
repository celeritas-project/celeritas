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

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/Environment.hh"

#if CELER_USE_DEVICE
#    include <thrust/system/system_error.h>
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
bool determine_strip()
{
    if (!celeritas::getenv("CELER_STRIP_SOURCEDIR").empty())
    {
        return true;
    }
    return static_cast<bool>(CELERITAS_DEBUG);
}

//---------------------------------------------------------------------------//
//! Try removing up to and including the filename from the reported path.
std::string strip_source_dir(const std::string& filename)
{
    static const bool do_strip = determine_strip();
    if (!do_strip)
    {
        // Don't strip in debug mode
        return filename;
    }

    std::string::size_type max_pos = 0;
    for (const std::string path : {"src/", "app/", "test/"})
    {
        auto pos = filename.rfind(path);

        if (pos != std::string::npos)
        {
            pos += path.size() - 1;
            max_pos = std::max(max_pos, pos);
        }
    }
    if (max_pos == 0)
    {
        // No telling where the filename is from...
        return filename;
    }

    return filename.substr(max_pos + 1);
}

//---------------------------------------------------------------------------//
} // namespace

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
            where << strip_source_dir(e.details().file);
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
        where << strip_source_dir(e.details().file) << ':' << e.details().line;
        std::ostringstream what;
        what << to_cstring(e.details().which) << ": " << e.details().condition;
        G4Exception(
            where.str().c_str(), err_code_, FatalException, what.str().c_str());
    }
#if CELER_USE_DEVICE
    catch (const thrust::system_error& e)
    {
        G4Exception("Thrust GPU library", err_code_, FatalException, e.what());
    }
#endif
    // (Any other errors will be rethrown and abort the program.)
}

//---------------------------------------------------------------------------//
} // namespace celeritas
