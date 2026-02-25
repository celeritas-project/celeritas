//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/PDFullSimCeler.hh
//! \note This file is an `art` plugin and should only be included by its .cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <art/Framework/Core/EDProducer.h>
#include <art/Utilities/ToolConfigTable.h>

#include "corecel/Macros.hh"
#include "celeritas/inp/StandaloneInput.hh"  // IWYU pragma: keep

#include "LarStandaloneRunner.hh"

#include "detail/PDFullSimCelerConfig.hh"

namespace sim
{
class SimEnergyDeposit;
class OpDetBacktrackerRecord;
}  // namespace sim

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Run optical photons in a standalone simulation.
 *
 * This plugin implements a replacement for LArSim's \c phot::PDFastSimPAR
 * class. It is instantiated by a FHiCL workflow file with a set of
 * parameters. It takes a vector of energy-depositing steps and returns a
 * vector of detector responses. It is executed after the detector simulation
 * module (ionization, recombination, scintillation, etc.) with a vector of
 * steps that contain local energy deposition.
 *
 * The execution happens \em after LArG4 is complete, so it is completely
 * independent of the Geant4 run manager and execution. It requires an input
 * GDML with:
 * - Detector geometry description
 * - Bulk optical physics properties (e.g., Rayleigh scattering in argon)
 * - Surface properties (e.g., roughness, reflection probability)
 * - Detector properties (e.g., sensitive volumes, efficiency multipliers)
 *
 * \internal
 * \par Parameter set definitions
 *
 * See \c celeritas::detail::PDFullSimCelerConfig .
 */
class PDFullSimCeler final : public art::EDProducer
{
  public:
    //!@{
    //! \name Type aliases
    using Config = detail::PDFullSimCelerConfig;
    using Parameters = art::EDProducer::Table<Config>;
    ///@}

  public:
    // Construct with fcl parameters
    explicit PDFullSimCeler(Parameters const& p);
    CELER_DELETE_COPY_MOVE(PDFullSimCeler);

    // Start simulating events
    void beginJob() final;

    // Simulate a single event
    void produce(art::Event&) final;

    // Tear down optical simulation library
    void endJob() final;

  private:
    //! Runner input for building in beginJob
    LarStandaloneRunner::Input runner_inp_;
    //! Identifying tag should usually be set to IonAndScint
    art::InputTag sim_tag_;

    //! Constructed runner to process an event
    std::unique_ptr<LarStandaloneRunner> runner_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
