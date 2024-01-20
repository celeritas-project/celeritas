//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/ScopedGeantLogger.cc
//---------------------------------------------------------------------------//
#include "ScopedGeantLogger.hh"

#include <cctype>
#include <memory>
#include <mutex>
#include <G4String.hh>
#include <G4Threading.hh>
#include <G4Types.hh>
#include <G4UImanager.hh>
#include <G4Version.hh>
#include <G4coutDestination.hh>
#if G4VERSION_NUMBER >= 1120
#    include <G4ios.hh>
#    define CELER_G4SSBUF 0
#else
#    include <G4strstreambuf.hh>
#    define CELER_G4SSBUF 1
#endif

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Log the actual message.
 */
G4int log_impl(G4String const& str, LogLevel level)
{
    // Output with dummy file/line
    ::celeritas::world_logger()({"Geant4", 0}, level) << trim(str);

    // 0 for success, -1 for failure
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * Send Geant4 log messages to Celeritas' world logger.
 */
class GeantLoggerAdapter : public G4coutDestination
{
  public:
    // Assign to Geant handlers on construction
    GeantLoggerAdapter();
    ~GeantLoggerAdapter();

    // Handle error messages
    G4int ReceiveG4cout(G4String const& str) final;
    G4int ReceiveG4cerr(G4String const& str) final;

  private:
#if CELER_G4SSBUF
    G4coutDestination* saved_cout_{nullptr};
    G4coutDestination* saved_cerr_{nullptr};
#endif
};

//---------------------------------------------------------------------------//
/*!
 * Redirect geant4's stdout/cerr on construction.
 */
GeantLoggerAdapter::GeantLoggerAdapter()
{
    if (!G4UImanager::GetUIpointer())
    {
        // Always-on debug assertion (not a "runtime" error but a
        // subtle programming logic error that always causes a crash)
        CELER_DEBUG_FAIL(
            "Geant4 logging cannot be changed after G4UImanager has been "
            "destroyed",
            precondition);
    }

#if CELER_G4SSBUF
    saved_cout_ = G4coutbuf.GetDestination();
    saved_cerr_ = G4cerrbuf.GetDestination();
    G4coutbuf.SetDestination(this);
    G4cerrbuf.SetDestination(this);
#else
    // See Geant4 change global-V11-01-01
    G4iosSetDestination(this);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Restore iostream buffers on destruction.
 */
GeantLoggerAdapter::~GeantLoggerAdapter()
{
#if CELER_G4SSBUF
    G4coutbuf.SetDestination(saved_cout_);
    G4cerrbuf.SetDestination(saved_cerr_);
#else
    G4iosSetDestination(nullptr);
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Process stdout message.
 */
G4int GeantLoggerAdapter::ReceiveG4cout(G4String const& str)
{
    return log_impl(str, LogLevel::diagnostic);
}

//---------------------------------------------------------------------------//
/*!
 * Process stderr message.
 */
G4int GeantLoggerAdapter::ReceiveG4cerr(G4String const& str)
{
    return log_impl(str, LogLevel::info);
}

//---------------------------------------------------------------------------//
//! Thread-local flag for "ownership" of the Geant4 logger
G4ThreadLocal bool g_adapter_active_{false};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Install the Celeritas Geant4 logger.
 *
 * A global flag allows multiple logger adapters to be nested without
 * consequence.
 */
ScopedGeantLogger::ScopedGeantLogger()
{
    if (!g_adapter_active_)
    {
        g_adapter_active_ = true;
        logger_ = std::make_unique<GeantLoggerAdapter>();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Revert to the previous exception handler.
 */
ScopedGeantLogger::~ScopedGeantLogger()
{
    if (logger_)
    {
        logger_.reset();
        g_adapter_active_ = false;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
