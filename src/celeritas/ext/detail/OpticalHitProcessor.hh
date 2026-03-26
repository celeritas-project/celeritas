//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/OpticalHitProcessor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>
#include <G4TouchableHandle.hh>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/DetectorData.hh"

class G4LogicalVolume;
class G4Step;
class G4StepPoint;
class G4Track;
class G4VSensitiveDetector;

namespace celeritas
{
class GeantGeoParams;

struct DetectorHitsOutput;

namespace optical
{
class CoreParams;
}  // namespace optical

namespace detail
{

//---------------------------------------------------------------------------//
// Build a vector of G4LogicalVolume* indexed by optical DetectorId
std::shared_ptr<std::vector<G4LogicalVolume const*> const>
build_optical_detector_volumes(optical::CoreParams const& opt_params);
class LevelTouchableUpdater;

//---------------------------------------------------------------------------//
/*!
 * Transfer Celeritas optical detector hits to Geant4 sensitive detectors.
 *
 * This class consumes a \c DetectorHitsOutput produced by
 * \c optical::DetectorAction and reconstructs a \c G4Step per hit with the
 * correct touchable (full volume hierarchy), then invokes
 * \c G4VSensitiveDetector::Hit() on the thread-local SD.
 *
 * \warning This class \b must be thread-local (same as \c HitProcessor):
 * Geant4 thread-local SD instances and allocators require that this object is
 * constructed and destroyed on the same thread it is used.
 */
class OpticalHitProcessor
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstVecLV
        = std::shared_ptr<std::vector<G4LogicalVolume const*> const>;
    using SPConstGeantGeo = std::shared_ptr<GeantGeoParams const>;
    //!@}

  public:
    // Construct with detector volumes and geometry
    OpticalHitProcessor(SPConstVecLV detector_volumes, SPConstGeantGeo geo);

    // Destroy (must be on the same thread as construction)
    ~OpticalHitProcessor();

    CELER_DEFAULT_MOVE_DELETE_COPY(OpticalHitProcessor);

    // Process a batch of optical hits (the DetectorAction callback target)
    void operator()(DetectorHitsOutput const& out);

    // Access thread-local SD for a detector ID
    inline G4VSensitiveDetector* detector(DetectorId) const;

    // Access detector logical volume for a detector ID
    inline G4LogicalVolume const* detector_volume(DetectorId) const;

  private:
    //! Detector volumes (indexed by DetectorId)
    SPConstVecLV detector_volumes_;
    //! Thread-local sensitive detectors (from LV->GetSensitiveDetector())
    std::vector<G4VSensitiveDetector*> detectors_;

    //! Shared G4Step object reused across hits
    std::shared_ptr<G4Step> step_;
    //! Post-step point (the only step point populated for optical photons)
    G4StepPoint* step_point_{nullptr};
    //! Touchable handle for the post-step point
    G4TouchableHandle touch_handle_;

    //! Single reused G4Track for G4OpticalPhoton
    G4Track* track_{nullptr};

    //! Touchable updater using volume instance hierarchy
    std::unique_ptr<LevelTouchableUpdater> update_touchable_;

    //// HELPER FUNCTIONS ////

    // Process a single hit
    void process_hit(optical::DetectorHit const& hit,
                     Span<VolumeInstanceId const> vol_span);
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access thread-local sensitive detector for a detector ID.
 */
inline G4VSensitiveDetector* OpticalHitProcessor::detector(DetectorId did) const
{
    CELER_EXPECT(did < detectors_.size());
    return detectors_[did.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access detector logical volume for a detector ID.
 */
inline G4LogicalVolume const*
OpticalHitProcessor::detector_volume(DetectorId did) const
{
    CELER_EXPECT(did < detector_volumes_->size());
    return (*detector_volumes_)[did.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
