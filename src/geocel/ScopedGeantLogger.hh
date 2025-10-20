//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/ScopedGeantLogger.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

class G4coutDestination;

namespace celeritas
{
class Logger;
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
    // Enable and disable to avoid recursion with accel/Logger
    static bool enabled();
    static void enabled(bool);

    // Construct with pointer to the Celeritas logger instance
    explicit ScopedGeantLogger(Logger&);

    // Construct using world logger
    ScopedGeantLogger();

    // Clear on destruction
    ~ScopedGeantLogger();

    //! Prevent copying and moving for RAII class
    CELER_DELETE_COPY_MOVE(ScopedGeantLogger);

  private:
#if CELERITAS_USE_GEANT4
    std::unique_ptr<G4coutDestination> logger_;
#endif
};

#if !CELERITAS_USE_GEANT4
// Do nothing if Geant4 is disabled (source file will not be compiled)
inline bool ScopedGeantLogger::enabled()
{
    return false;
}
inline ScopedGeantLogger::ScopedGeantLogger(Logger&) {}
inline ScopedGeantLogger::ScopedGeantLogger() {}
inline ScopedGeantLogger::~ScopedGeantLogger() {}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
