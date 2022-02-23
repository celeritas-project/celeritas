//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeantExceptionHandler.cc
//---------------------------------------------------------------------------//
#include "GeantExceptionHandler.hh"

#include <sstream>

#include "base/Assert.hh"
#include "comm/Logger.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate exceptions to Celeritas.
 */
G4bool GeantExceptionHandler::Notify(const char*         origin_of_exception,
                                     const char*         exception_code,
                                     G4ExceptionSeverity severity,
                                     const char*         description)
{
    CELER_EXPECT(origin_of_exception);
    CELER_EXPECT(exception_code);

    // Construct message
    std::ostringstream os;
    os << exception_code << " in Geant4 " << origin_of_exception << ": "
       << description;
    const std::string& msg = os.str();

    switch (severity)
    {
        case FatalException:
        case FatalErrorInArgument:
        case RunMustBeAborted:
        case EventMustBeAborted:
            // Severe or initialization error
            throw celeritas::RuntimeError(msg);
        case JustWarning:
            // Display a message
            CELER_LOG(warning) << msg;
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    // Return "true" to cause Geant4 to crash the program, or "false" to let it
    // know that we've handled the exception.
    return false;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
