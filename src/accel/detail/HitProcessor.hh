//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <G4TouchableHandle.hh>

#include "celeritas/Types.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

class G4LogicalVolume;
class G4Step;
class G4Navigator;
class G4Track;
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
 * generating hit objects. It \b must be thread-local because the sensitive
 * detectors it stores are thread-local, and additionally Geant4 makes
 * assumptions about object allocations that cause crashes if the HitProcessor
 * is allocated on one thread and destroyed on another.
 *
 * Call operator:
 * - Loop over detector steps
 * - Update step attributes based on hit selection for the detector (TODO:
 *   selection is global for now)
 * - Call the local detector (based on detector ID from map) with the step
 */
class HitProcessor
{
  public:
    //!@{
    //! \name Type aliases
    using StepStateHostRef = HostRef<StepStateData>;
    using StepStateDeviceRef = DeviceRef<StepStateData>;
    using SPConstVecLV
        = std::shared_ptr<const std::vector<G4LogicalVolume const*>>;
    //!@}

  public:
    // Construct from volumes that have SDs and step selection
    HitProcessor(SPConstVecLV detector_volumes,
                 StepSelection const& selection,
                 bool locate_touchable);

    // Default destructor
    ~HitProcessor();

    // Process CPU-generated hits
    void operator()(StepStateHostRef const&);

    // Process device-generated hits
    void operator()(StepStateDeviceRef const&);

    // Generate and call hits from a detector output (for testing)
    void operator()(DetectorStepOutput const& out) const;

  private:
    //! Detector volumes for navigation updating
    SPConstVecLV detector_volumes_;
    //! Map detector IDs to sensitive detectors
    std::vector<G4VSensitiveDetector*> detectors_;
    //! Temporary CPU hit information
    DetectorStepOutput steps_;

    //! Temporary step
    std::unique_ptr<G4Step> step_;
    //! Temporary track, required by some frameworks
    std::unique_ptr<G4Track> track_;
    //! Navigator for finding points
    std::unique_ptr<G4Navigator> navi_;
    //! Geant4 reference-counted pointer to a G4VTouchable
    G4TouchableHandle touch_handle_;

    bool update_touchable(Real3 const& pos,
                          Real3 const& dir,
                          G4LogicalVolume const* lv) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
