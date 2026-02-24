//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.hh
//! \note This file is an `art` plugin and should only be included by its .cc
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <art/Utilities/ToolConfigTable.h>
#include <larsim/PhotonPropagation/OpticalPropagationTools/IOpticalPropagation.h>

#include "LarStandaloneRunner.hh"

#include "detail/LarCelerConfig.hh"

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
 * See \c celeritas::detail::LarCelerStandaloneConfig .
 */
class LarCelerStandalone final : public phot::IOpticalPropagation
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
    LarStandaloneRunner::Input runner_inp_;
    std::unique_ptr<LarStandaloneRunner> runner_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
