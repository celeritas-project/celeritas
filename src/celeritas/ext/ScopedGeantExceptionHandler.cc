//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/ScopedGeantExceptionHandler.cc
//---------------------------------------------------------------------------//
#include "ScopedGeantExceptionHandler.hh"

#include <G4ExceptionSeverity.hh>
#include <G4StateManager.hh>
#include <G4Types.hh>
#include <G4VExceptionHandler.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Process Geant4 exceptions with Celeritas.
 *
 * The Geant exception handler base class changes global state in its
 * constructor (assigning "this") so this class instance must stay in scope
 * once created. There is no way to save or restore the previous handler.
 * Furthermore, creating a G4RunManagerKernel also resets the exception
 * handler, so errors thrown during setup *CANNOT* be caught by celeritas, and
 * this class can only be used after creating the G4RunManager.
 */
class GeantExceptionHandler final : public G4VExceptionHandler
{
  public:
    // Accept error codes from geant4
    G4bool Notify(char const* originOfException,
                  char const* exceptionCode,
                  G4ExceptionSeverity severity,
                  char const* description) final;
};

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
            CELER_LOG_LOCAL(error) << err.what();
            break;
        default:
            CELER_ASSERT_UNREACHABLE();
    }

    // Return "true" to cause Geant4 to crash the program, or "false" to let it
    // know that we've handled the exception.
    return false;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Install the Celeritas Geant4 exception handler.
 *
 * The base class of the exception handler calls SetExceptionHandler...
 */
ScopedGeantExceptionHandler::ScopedGeantExceptionHandler()
{
    auto* state_mgr = G4StateManager::GetStateManager();
    CELER_ASSERT(state_mgr);
    previous_ = state_mgr->GetExceptionHandler();
    current_ = std::make_unique<GeantExceptionHandler>();
    CELER_ENSURE(state_mgr->GetExceptionHandler() == current_.get());
}

//---------------------------------------------------------------------------//
/*!
 * Revert to the previous exception handler.
 */
ScopedGeantExceptionHandler::~ScopedGeantExceptionHandler()
{
    auto* state_mgr = G4StateManager::GetStateManager();
    if (state_mgr->GetExceptionHandler() == current_.get())
    {
        state_mgr->SetExceptionHandler(previous_);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
