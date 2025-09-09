//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserRunAction.hh>

#include "corecel/sys/Stopwatch.hh"
#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"

class G4VExceptionHandler;

namespace celeritas
{
class ScopedGeantLogger;
class ScopedGeantExceptionHandler;

namespace app
{
class GeantDiagnostics;
//---------------------------------------------------------------------------//
/*!
 * Set up and tear down Celeritas.
 *
 * Each Geant4 thread creates an instance of this class. In multithreaded mode,
 * the "master" instance does not have a local transporter and is responsible
 * for initializing the \c SharedParams which is shared across all
 * threads/tasks. Worker threads are given a thread-local \c
 * LocalTransporter which allocates Celeritas track state data at
 * the beginning of the run and clears it at the end.
 */
class RunAction final : public G4UserRunAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstOptions = std::shared_ptr<SetupOptions const>;
    using SPParams = std::shared_ptr<SharedParams>;
    using SPTransporter = std::shared_ptr<LocalTransporter>;
    using SPDiagnostics = std::shared_ptr<GeantDiagnostics>;
    //!@}

  public:
    RunAction(SPConstOptions options,
              SPParams params,
              SPTransporter transport,
              SPDiagnostics diagnostics,
              bool init_shared);

    void BeginOfRunAction(G4Run const* run) final;
    void EndOfRunAction(G4Run const* run) final;

  private:
    SPConstOptions options_;
    SPParams params_;
    SPTransporter transport_;
    SPDiagnostics diagnostics_;
    bool init_shared_;
    Stopwatch get_transport_time_;

    // Thread-local Geant4 logging and exception redirect
    std::unique_ptr<ScopedGeantLogger> scoped_log_;
    std::shared_ptr<ScopedGeantExceptionHandler> local_eh_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
