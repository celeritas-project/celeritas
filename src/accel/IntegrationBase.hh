//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationBase.hh
//---------------------------------------------------------------------------//
#pragma once

class G4Run;
#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct SetupOptions;

//---------------------------------------------------------------------------//
/*!
 * Common interface for integrating Celeritas into user applications.
 *
 * This implements common functionality for the Celeritas integration classes.
 *
 * \sa celeritas::UserActionIntegration
 * \sa celeritas::TrackingManagerIntegration
 *
 * \note For developers: this and the integration daughters all share common
 * data in detail::IntegrationSingleton.
 */
class IntegrationBase
{
  public:
    // Edit options before starting the run
    void SetOptions(SetupOptions&& opts);

    // Initialize during ActionInitialization on non-worker thread
    void BuildForMaster();

    // Initialize during ActionInitialization
    void Build();

    // Start the run
    virtual void BeginOfRunAction(G4Run const* run) = 0;

    // End the run
    void EndOfRunAction(G4Run const* run);

  protected:
    IntegrationBase() = default;
    ~IntegrationBase() = default;
    CELER_DEFAULT_COPY_MOVE(IntegrationBase);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
