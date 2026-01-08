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

namespace celeritas
{
class Logger;
//---------------------------------------------------------------------------//
/*!
 * Redirect Geant4 logging through Celeritas' logger.
 *
 * This parses messages sent to \c G4cout and \c G4cerr from Geant4. Based on
 * the message (whether it starts with warning, error, '!!!', '***') it tries
 * to use the appropriate logging level and source context.
 *
 * Since the Geant4 output streams are thread-local, this class is as well.
 * Multiple geant loggers can be nested in scope, but only the outermost on a
 * given thread will "own" the log destination.
 *
 * - When instantiated during setup, this should be constructed with
 *   \c celeritas::world_logger to avoid printing duplicate messages per
 *   thread/process.
 * - When instantiated during runtime, it should take the
 *   \c celeritas::self_logger so that only warning/error messages are printed
 *   for event/track-specific details.
 *
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
    class StreamDestination;

    std::unique_ptr<StreamDestination> logger_;
};

#if !CELERITAS_USE_GEANT4
// Do nothing if Geant4 is disabled
inline bool ScopedGeantLogger::enabled()
{
    return false;
}
inline void ScopedGeantLogger::enabled(bool) {}
inline ScopedGeantLogger::ScopedGeantLogger(Logger&) {}
inline ScopedGeantLogger::ScopedGeantLogger() {}
inline ScopedGeantLogger::~ScopedGeantLogger() {}
class ScopedGeantLogger::StreamDestination
{
};
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
