//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/LarCelerStandalone.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

namespace fhicl
{
class ParameterSet;
}
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
 * class, taking a vector of energy-depositing steps and returning a vector
 * is instantiated by a FHICL workflow file with a set of
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
 * \par Parameter set definitions
 *
 * To be defined later, but we will need:
 * - GDML input filename
 * - Performance tweaking knobs (e.g., number of tracks in flight)
 * - ...
 */
class LarCelerStandalone
{
  public:
    //!@{
    //! \name Type aliases
    using VecSED = std::vector<sim::SimEnergyDeposit>;
    using VecBTR = std::vector<sim::OpDetBacktrackerRecord>;
    using UPVecBTR = std::unique_ptr<VecBTR>;
    ///@}

    // Construct with fcl parameters
    LarCelerStandalone(fhicl::ParameterSet const& p);

    // Execute simulation
    UPVecBTR execute(VecSED const& edeps);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
