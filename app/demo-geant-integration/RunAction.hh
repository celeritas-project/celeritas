//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geant-integration/RunAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserRunAction.hh>

#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"

namespace demo_geant
{
//---------------------------------------------------------------------------//
/*!
 * Set up and tear down Celeritas.
 *
 * Each Geant4 thread creates an instance of this class. In multithreaded mode,
 * the "master" instance does not have a local transporter and is responsible
 * for initializing the \c celeritas::SharedParams which is shared across all
 * threads/tasks. Worker threads are given a thread-local \c
 * celeritas::LocalTransporter which allocates Celeritas track state data at
 * the beginning of the run and clears it at the end.
 */
class RunAction final : public G4UserRunAction
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstOptions = std::shared_ptr<const celeritas::SetupOptions>;
    using SPParams = std::shared_ptr<celeritas::SharedParams>;
    using SPTransporter = std::shared_ptr<celeritas::LocalTransporter>;
    //!@}

  public:
    RunAction(SPConstOptions options,
              SPParams params,
              SPTransporter transport,
              bool init_celeritas);

    void BeginOfRunAction(G4Run const* run) final;
    void EndOfRunAction(G4Run const* run) final;

  private:
    SPConstOptions options_;
    SPParams params_;
    SPTransporter transport_;
    bool init_celeritas_;
};

//---------------------------------------------------------------------------//
}  // namespace demo_geant
