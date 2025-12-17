//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.hh
//! \note This file is an `art` plugin and should only be included by its .cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <art/Utilities/ToolConfigTable.h>

#include "LarStandaloneRunner.hh"

#include "detail/LarCelerConfig.hh"

namespace sim
{
class SimEnergyDeposit;
class OpDetBacktrackerRecord;
}  // namespace sim

// TODO: This will be defined upstream:
// see https://github.com/nuRiceLab/larsim/pull/1
namespace phot
{
class OpticalSimInterface
{
  public:
    //!@{
    //! \name Type aliases
    using VecSED = std::vector<sim::SimEnergyDeposit>;
    using VecBTR = std::vector<sim::OpDetBacktrackerRecord>;
    using UPVecBTR = std::unique_ptr<VecBTR>;
    ///@}

    // Enable polymorphic deletion
    virtual ~OpticalSimInterface() = 0;

    // Set up execution
    virtual void beginJob() = 0;

    // Process a single event, returning detector hits
    virtual UPVecBTR executeEvent(VecSED const& edeps) = 0;

    // Tear down execution
    virtual void endJob() = 0;
};
}  // namespace phot

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Run optical photons in a standalone simulation.
 *
 * This plugin implements a replacement for LArSim's \c phot::PDFastSimPAR
 * class, taking a vector of energy-depositing steps and returning a vector
 * is instantiated by a FHiCL workflow file with a set of
 * parameters. It is executed after the detector simulation step (ionization,
 * recombination, scintillation, etc.) with a vector of steps that contain
 * energy deposition, and it returns a vector of detector responses.
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
 * See \c celeritas::detail::LarCelerStandaloneConfig .
 */
class LarCelerStandalone final : public phot::OpticalSimInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Config = detail::LarCelerStandaloneConfig;
    using Parameters = art::ToolConfigTable<Config>;
    ///@}

  public:
    // Construct with fcl parameters
    LarCelerStandalone(Parameters const& p);

    // Start simulating events
    void beginJob() final;

    // Simulate a single event
    UPVecBTR executeEvent(VecSED const& edeps) final;

    // Complete the simulation
    void endJob() final;

  private:
    inp::LarStandaloneRunner runner_inp_;
    std::unique_ptr<LarStandaloneRunner> runner_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
