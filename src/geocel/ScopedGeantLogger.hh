//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ScopedGeantLogger.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas_config.h"
#include "corecel/Macros.hh"

class G4coutDestination;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Install a Geant output destination during this class's lifetime.
 *
 * Since the Geant4 output streams are thread-local, this class is as well.
 * Multiple geant loggers can be nested, and only the outermost on a given
 * thread will "own" the log destination.
 */
class ScopedGeantLogger
{
  public:
    // Construct exception handler
    ScopedGeantLogger();

    // Clear on destruction
    ~ScopedGeantLogger();
    //!@{
    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedGeantLogger);
    //!@}

  private:
#if CELERITAS_USE_GEANT4
    std::unique_ptr<G4coutDestination> logger_;
#endif
};

#if !CELERITAS_USE_GEANT4
//!@{
//! Do nothing if Geant4 is disabled (source file will not be compiled)
inline ScopedGeantLogger::ScopedGeantLogger() {}
inline ScopedGeantLogger::~ScopedGeantLogger() {}
//!@}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
