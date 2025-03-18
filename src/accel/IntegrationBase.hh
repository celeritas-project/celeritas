//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

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
 * Celeritas offloading enabled.
 *
 * \sa celeritas::UserActionIntegration
 * \sa celeritas::TrackingManagerIntegration
 *
 * \note For developers: this and the integration daughters all share common
 * data in \c detail::IntegrationSingleton.
 */
class IntegrationBase
{
  public:
    // Set options before starting the run
    void SetOptions(SetupOptions&& opts);

    // Initialize during ActionInitialization on non-worker thread
    void BuildForMaster();

    // Initialize during ActionInitialization
    void Build();

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
    IntegrationBase() = default;
    ~IntegrationBase() = default;
    CELER_DEFAULT_COPY_MOVE(IntegrationBase);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
