//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/ScopedGeantExceptionHandler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"

class G4VExceptionHandler;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Install and clear a Geant exception handler during this class lifetime.
 *
 * Note that creating a \c G4RunManagerKernel resets the exception
 * handler, so errors thrown during setup *CANNOT* be caught by Celeritas, and
 * this class can only be used after creating the \c G4RunManager.
 */
class ScopedGeantExceptionHandler
{
  public:
    // Construct exception handler
    ScopedGeantExceptionHandler();

    // Clear on destruction
    ~ScopedGeantExceptionHandler();
    //!@{
    //! no move; no copying
    ScopedGeantExceptionHandler(ScopedGeantExceptionHandler const&) = delete;
    ScopedGeantExceptionHandler& operator=(ScopedGeantExceptionHandler const&)
        = delete;
    ScopedGeantExceptionHandler(ScopedGeantExceptionHandler&&) = delete;
    ScopedGeantExceptionHandler& operator=(ScopedGeantExceptionHandler&&)
        = delete;
    //!@}

  private:
#if CELERITAS_USE_GEANT4
    G4VExceptionHandler* previous_{nullptr};
    std::unique_ptr<G4VExceptionHandler> current_;
#endif
};

#if !CELERITAS_USE_GEANT4
//!@{
//! Do nothing if Geant4 is disabled (source file will not be compiled)
inline ScopedGeantExceptionHandler::ScopedGeantExceptionHandler() {}
inline ScopedGeantExceptionHandler::~ScopedGeantExceptionHandler() {}
//!@}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
