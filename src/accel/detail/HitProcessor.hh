//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <G4Navigator.hh>
#include <G4Step.hh>
#include <G4TouchableHandle.hh>

#include "celeritas/Types.hh"

class G4VSensitiveDetector;

namespace celeritas
{
struct StepSelection;
struct DetectorStepOutput;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Transfer Celeritas sensitive detector hits to Geant4.
 *
 * This serves a similar purpose to the \c G4FastSimHitMaker class for
 * generating hit objects. It should be "stream local" but by necessity has to
 * share the same (TODO: name?) HitProcessorManager which takes the input
 * detector names, maps them to detector IDs, and adds a HitCollector to the
 * action manager.
 *
 * Manager:
 * - Find logical volumes (global!) with attached SDs (local!); SDManager
 *   doesn't have good support for iterating over detectors
 *
 * Local:
 * - Get local G4SD for each G4LV
 * - Loop over detector steps
 * - Map detector ID to local sensitive detector
 * - Update step attributes based on hit selection for the detector (TODO:
 *   selection is global for now)
 * - Call the local detector with the step
 */
class HitProcessor
{
  public:
    //!@{
    //! \name Type aliases
    using VecLV = std::vector<G4LogicalVolume*>;
    //!@}

  public:
    // Construct from volumes that have SDs and step selection
    HitProcessor(const VecLV& detector_volumes, const StepSelection& selection);

    // Generate and call hits from a detector output
    void operator()(const DetectorStepOutput& out) const;

  private:
    //! Temporary step
    std::unique_ptr<G4Step> step_;
    //! Navigator for finding points
    std::unique_ptr<G4Navigator> navi_;
    //! Geant4 reference-counted pointer to a G4VTouchable
    G4TouchableHandle touch_handle_;
    //! Map detector IDs to sensitive detector pointers
    std::vector<G4VSensitiveDetector*> detectors_;
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
