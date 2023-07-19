//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RunAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <G4UserRunAction.hh>

#include "corecel/io/OutputRegistry.hh"
#include "accel/LocalTransporter.hh"
#include "accel/SetupOptions.hh"
#include "accel/SharedParams.hh"

#include "TrackStepCounter.hh"

namespace celeritas
{
namespace app
{
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
    using SPStepCounter = std::shared_ptr<TrackStepCounter>;
    using SPTransporter = std::shared_ptr<LocalTransporter>;
    using SPOutputRegistry = std::shared_ptr<OutputRegistry>;
    //!@}

  public:
    RunAction(SPConstOptions options,
              SPParams params,
              SPTransporter transport,
              SPStepCounter step_counter,
              bool init_celeritas,
              bool init_diagnostics);

    void BeginOfRunAction(G4Run const* run) final;
    void EndOfRunAction(G4Run const* run) final;

  private:
    SPConstOptions options_;
    SPParams params_;
    SPTransporter transport_;
    SPStepCounter step_counter_;
    bool init_celeritas_;
    bool init_diagnostics_;
    bool disable_offloading_;
    SPOutputRegistry output_reg_;
};

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
