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
 * Because the underlying Geant4 error handler is thread-local, this class must
 * be scoped to inside each worker thread. Additionally, since throwing from a
 * worker thread terminates the program, an error handler \em must be specified
 * if used in a worker thread: you should probably use a \c
 * celeritas::MultiExceptionHandler if used inside a worker thread.
 *
 * \note Creating a \c G4RunManagerKernel resets the exception
 * handler, so errors thrown during setup \em cannot be caught by Celeritas,
 * and this class can only be used after creating the \c G4RunManager.
 */
class ScopedGeantExceptionHandler
{
  public:
    //!@{
    //! \name Type aliases
    using StdExceptionHandler = std::function<void(std::exception_ptr)>;
    //!@}

  public:
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
#endif
};

#if !CELERITAS_USE_GEANT4
//!@{
//! Do nothing if Geant4 is disabled (source file will not be compiled)
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
