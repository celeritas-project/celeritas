//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/ExceptionHandler.cc
//---------------------------------------------------------------------------//
#include "ExceptionHandler.hh"

#include <G4ExceptionSeverity.hh>
#include <G4RunManager.hh>
#include <G4StateManager.hh>
#include <G4Types.hh>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an exception handler that can catch exceptions.
 */
ExceptionHandler::ExceptionHandler(StdExceptionHandler handle_exception)
    : handle_{std::move(handle_exception)}
{
    CELER_EXPECT(handle_);
}

//---------------------------------------------------------------------------//
/*!
 * Raise an exception, catch it with the handler, and abort.
 */
G4bool ExceptionHandler::Notify(char const* origin_of_exception,
                                char const* exception_code,
                                G4ExceptionSeverity severity,
                                char const* description)
{
    CELER_EXPECT(origin_of_exception);
    CELER_EXPECT(exception_code);

    // Construct message
    auto err = RuntimeError::from_geant_exception(
        origin_of_exception, exception_code, description);
    bool must_abort{false};

    switch (severity)
    {
        case FatalException:
        case FatalErrorInArgument:
        case RunMustBeAborted:
        case EventMustBeAborted:
            CELER_TRY_HANDLE(throw err, handle_);
            if (auto* run_man = G4RunManager::GetRunManager())
            {
                if (severity == EventMustBeAborted)
                {
                    run_man->AbortEvent();
                }
                else
                {
                    run_man->AbortRun(/* softAbort = */ true);
                }
            }
            else
            {
                must_abort = true;
            }
            break;
        case JustWarning:
            // Display a message
            CELER_LOG_LOCAL(error) << err.what();
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    // Return "true" to cause Geant4 to crash the program, or "false" to let it
    // know that we've handled the exception.
    return must_abort;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
