//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

#include "Types.hh"

class G4Run;

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SetupOptions;
class CoreStateInterface;
class CoreParams;

//---------------------------------------------------------------------------//
/*!
 * Common interface for integrating Celeritas into user applications.
 *
 * This implements common functionality for the Celeritas integration classes.
 * The \c GetParams and \c GetState methods may only be used during a run with
 * Celeritas offloading enabled. It cannot be accessed before the run manager
 * is created (this requirement may be relaxed in the future).
 *
 * \sa celeritas::UserActionIntegration
 * \sa celeritas::TrackingManagerIntegration
 *
 * \internal This and the integration daughters all share common
 * data in \c detail::IntegrationSingleton.
 */
class IntegrationBase
{
  public:
    // Access whether celeritas is set up, enabled, or uninitialized
    OffloadMode GetMode() const;

    // Set options before starting the run
    void SetOptions(SetupOptions&& opts);

    // REMOVE in v0.7
    [[deprecated]] void BuildForMaster() {}

    // REMOVE in v0.7
    [[deprecated]] void Build() {}

    // Start the run
    virtual void BeginOfRunAction(G4Run const* run) = 0;

    // End the run
    void EndOfRunAction(G4Run const* run);

    //// ACCESSORS ////

    // Access Celeritas shared params
    CoreParams const& GetParams();

    // Access THREAD-LOCAL Celeritas core state data for user diagnostics
    CoreStateInterface& GetState();

  protected:
    IntegrationBase();
    ~IntegrationBase() = default;
    CELER_DEFAULT_COPY_MOVE(IntegrationBase);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
