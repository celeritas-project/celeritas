//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantExceptionHandler.cc
//---------------------------------------------------------------------------//
#include "GeantExceptionHandler.hh"

#include <memory>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Propagate exceptions to Celeritas.
 */
G4bool GeantExceptionHandler::Notify(char const* origin_of_exception,
                                     char const* exception_code,
                                     G4ExceptionSeverity severity,
                                     char const* description)
{
    CELER_EXPECT(origin_of_exception);
    CELER_EXPECT(exception_code);

    // Construct message
    auto err = RuntimeError::from_geant_exception(
        origin_of_exception, exception_code, description);

    switch (severity)
    {
        case FatalException:
        case FatalErrorInArgument:
        case RunMustBeAborted:
        case EventMustBeAborted:
            // Severe or initialization error
            throw err;
        case JustWarning:
            // Display a message
            CELER_LOG(warning) << err.what();
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    // Return "true" to cause Geant4 to crash the program, or "false" to let it
    // know that we've handled the exception.
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
