//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ScopedGeantExceptionHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <exception>
#include <functional>
#include <memory>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"

class G4VExceptionHandler;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Convert Geant4 exceptions to RuntimeError during this class lifetime.
 *
 * Note that creating a \c G4RunManagerKernel resets the exception
 * handler, so errors thrown during setup \em cannot be caught by Celeritas,
 * and this class can only be used after creating the \c G4RunManager.
 *
 * Because the underlying Geant4 error handler is thread-local, this class must
 * be scoped to inside each worker thread. Additionally, since throwing from a
 * worker thread terminates the program, an error handler \em must be specified
 * if used in a worker thread: you should probably use a \c
 * celeritas::MultiExceptionHandler .
 *
 * A severity level of \c JustWarning to a \c G4Exception call will result in a
 * warning message being logged; otherwise, the given exception handler will be
 * called. (If not provided, the default behavior is to throw.) If the
 * exception handler returns without throwing, then the \em global \c
 * suppressed_fatal flag is set, and if a simulation is in progress, \c
 * G4RunManager::AbortRun() will be called. Note that if an error is thrown
 * during \c BeginRun, the first event \em will be started (in MT mode, on each
 * worker thread).
 */
class ScopedGeantExceptionHandler
{
  public:
    //!@{
    //! \name Type aliases
    using StdExceptionHandler = std::function<void(std::exception_ptr)>;
    //!@}

  public:
    // Whether a fatal exception call did not produce a thrown exception
    static bool suppressed_fatal();

    // Construct with an exception handling function
    explicit ScopedGeantExceptionHandler(StdExceptionHandler handle);

    //! Construct, throwing on G4Exception calls
    ScopedGeantExceptionHandler() : ScopedGeantExceptionHandler{nullptr} {}

    // Clear on destruction
    ~ScopedGeantExceptionHandler();

    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedGeantExceptionHandler);

  private:
#if CELERITAS_USE_GEANT4
    G4VExceptionHandler* previous_{nullptr};
    std::unique_ptr<G4VExceptionHandler> current_;
    bool suppressed_fatal_{false};
#endif
};

#if !CELERITAS_USE_GEANT4
//!@{
//! Do nothing if Geant4 is disabled (source file will not be compiled)
inline bool ScopedGeantExceptionHandler::suppressed_fatal()
{
    return false;
}
inline ScopedGeantExceptionHandler::ScopedGeantExceptionHandler(
    StdExceptionHandler)
{
    CELER_NOT_CONFIGURED("Geant4");
}
inline ScopedGeantExceptionHandler::~ScopedGeantExceptionHandler() {}
//!@}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
