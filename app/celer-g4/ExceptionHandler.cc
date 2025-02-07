//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "accel/SharedParams.hh"

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an exception handler that can catch exceptions.
 */
ExceptionHandler::ExceptionHandler(StdExceptionHandler handle_exception,
                                   SPConstParams params)
    : handle_{std::move(handle_exception)}, params_{std::move(params)}
{
    CELER_EXPECT(handle_);
    CELER_EXPECT(params_);
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
    auto err = RuntimeError{[&] {
        RuntimeErrorDetails details;
        details.which = "Geant4";
        details.what = description;
        details.condition = exception_code;
        details.file = origin_of_exception;
        return details;
    }()};

    bool must_abort{false};

    switch (severity)
    {
        case FatalException:
        case FatalErrorInArgument:
        case RunMustBeAborted:
        case EventMustBeAborted:
            // Propagate error message into the handler
            CELER_TRY_HANDLE(throw err, handle_);
            CELER_LOG_LOCAL(debug) << err.what();
            if (auto* run_man = G4RunManager::GetRunManager())
            {
                if (severity == EventMustBeAborted
                    && (params_->mode() != SharedParams::Mode::enabled))
                {
                    // Event can only be aborted if Celeritas is disabled
                    // because we can't clear the track buffer
                    CELER_LOG_LOCAL(error)
                        << R"(Aborting event due to exception)";
                    run_man->AbortEvent();
                }
                else
                {
                    CELER_LOG_LOCAL(critical)
                        << "Aborting run due to exception (" << exception_code
                        << ")";
                    run_man->AbortRun();
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
