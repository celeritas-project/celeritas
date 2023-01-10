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
#include <vector>
#include <G4TouchableHandle.hh>

#include "celeritas/Types.hh"

class G4LogicalVolume;
class G4Step;
class G4Navigator;
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
 * share the same HitManager which takes the input
 * detector names, maps them to detector IDs, and adds a HitCollector to the
 * action manager.
 *
 * Call operator:
 * - Loop over detector steps
 * - Update step attributes based on hit selection for the detector (TODO:
 *   selection is global for now)
 * - Call the local detector (based on detector ID from map) with the step
 *
 * \note For now we store the LogicalVolume rather than the SD in order to work
 * better with multithreaded code. (The LV `GetSensitiveDetector` returns
 * thread-local data.) This means you can (and should) share the \c
 * HitProcessor instance across threads. Once multithreading is better
 * integrated into Celeritas and we can check how it interacts with CMSSW, then
 * we can make the whole kernel thread safe by ensuring independent hit
 * processors (and state data) for every thread.
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
    HitProcessor(VecLV detector_volumes,
                 StepSelection const& selection,
                 bool locate_touchable);

    // Default destructor
    ~HitProcessor();

    // Generate and call hits from a detector output
    void operator()(DetectorStepOutput const& out) const;

  private:
    //! Map detector IDs to logical volumes
    VecLV detector_volumes_;
    //! Temporary step
    std::unique_ptr<G4Step> step_;
    //! Navigator for finding points
    std::unique_ptr<G4Navigator> navi_;
    //! Geant4 reference-counted pointer to a G4VTouchable
    G4TouchableHandle touch_handle_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
