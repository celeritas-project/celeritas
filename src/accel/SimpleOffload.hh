//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/SimpleOffload.hh
//---------------------------------------------------------------------------//
#pragma once

class G4Run;
class G4Event;
class G4Track;

namespace celeritas
{
class SharedParams;
struct SetupOptions;
class LocalTransporter;

//---------------------------------------------------------------------------//
/*!
 * Compressed interface for running Celeritas in a multithread Geant4 app.
 *
 * This class *must* be a thread-local instance with references to data that
 * exceed the lifetime of the class: e.g. SharedParams can be a global
 * variable, and LocalTransporter can be a global variable with \c thread_local
 * storage duration.
 *
 * The \c CELER_DISABLE environment variable, if set and non-empty, will
 * disable offloading so that Celeritas will not be built nor kill tracks.
 *
 * The method names correspond to methods in Geant4 User Actions and *must* be
 * called from all threads, both worker and master.
 */
class SimpleOffload
{
  public:
    //! Construct with celeritas disabled
    SimpleOffload() = default;

    // Construct from a reference to shared params and local data
    SimpleOffload(SetupOptions const* setup,
                  SharedParams* params,
                  LocalTransporter* local);

    //! Lazy initialization of this class on a worker thread
    void Build(SetupOptions const* setup,
               SharedParams* params,
               LocalTransporter* local)
    {
        *this = {setup, params, local};
    }

    //! Lazy initialization of this class on the master thread
    void BuildForMaster(SetupOptions const* setup, SharedParams* params)
    {
        *this = {setup, params, nullptr};
    }

    // Initialize celeritas data from setup options
    void BeginOfRunAction(G4Run const* run);

    // Send Celeritas the event ID
    void BeginOfEventAction(G4Event const* event);

    // Send tracks to Celeritas if applicable and "StopAndKill" if so
    void PreUserTrackingAction(G4Track* track);

    // Flush offloaded tracks from Celeritas
    void EndOfEventAction(G4Event const* event);

    // Finalize
    void EndOfRunAction(G4Run const* run);

    // Whether offloading is enabled
    explicit operator bool() const;

  private:
    SetupOptions const* setup_{nullptr};
    SharedParams* params_{nullptr};
    LocalTransporter* local_{nullptr};
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
