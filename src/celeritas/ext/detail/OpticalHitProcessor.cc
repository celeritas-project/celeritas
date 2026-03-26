//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/OpticalHitProcessor.cc
//---------------------------------------------------------------------------//
#include "OpticalHitProcessor.hh"

#include <limits>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4OpticalPhoton.hh>
#include <G4ParticleTable.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4VSensitiveDetector.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "geocel/DetectorParams.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/DetectorData.hh"

#include "LevelTouchableUpdater.hh"
#include "OpticalHitProcessorRegistry.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Build a vector of G4LogicalVolume* indexed by optical DetectorId.
 *
 * Iterates over all G4LogicalVolumes with attached SDs and maps them to
 * optical DetectorIds via GeantGeoParams and the optical DetectorParams.
 * Returns null if no optical detector params are available.
 */
std::shared_ptr<std::vector<G4LogicalVolume const*> const>
build_optical_detector_volumes(optical::CoreParams const& opt_params)
{
    auto const& det_params = opt_params.detectors();
    if (!det_params || det_params->num_detectors() == 0)
    {
        return nullptr;
    }

    auto geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geo,
                   << "GeantGeoParams not available while building optical "
                      "detector volume map");

    auto lv_vec = std::make_shared<std::vector<G4LogicalVolume const*>>(
        det_params->num_detectors(), nullptr);

    for (G4LogicalVolume const* lv : *G4LogicalVolumeStore::GetInstance())
    {
        if (!lv || !lv->GetSensitiveDetector())
            continue;

        VolumeId vol_id = geo->geant_to_id(*lv);
        if (!vol_id)
            continue;

        DetectorId det_id = det_params->detector_id(vol_id);
        if (!det_id)
            continue;

        CELER_ASSERT(det_id.unchecked_get() < lv_vec->size());
        (*lv_vec)[det_id.unchecked_get()] = lv;
    }

    // Validate: every detector slot must have an LV
    for (auto i : range(lv_vec->size()))
    {
        CELER_VALIDATE((*lv_vec)[i],
                       << "no G4LogicalVolume found for optical detector "
                       << i);
    }

    return lv_vec;
}

//---------------------------------------------------------------------------//
/*!
 * Return a reference to the thread-local optical hit processor pointer.
 */
OpticalHitProcessor*& thread_local_optical_hit_processor()
{
    thread_local OpticalHitProcessor* ptr = nullptr;
    return ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Construct thread-local optical hit processor.
 *
 * Must be called on the Geant4 worker thread that will use this object.
 */
OpticalHitProcessor::OpticalHitProcessor(SPConstVecLV detector_volumes,
                                         SPConstGeantGeo geo)
    : detector_volumes_(std::move(detector_volumes))
    , step_{std::make_shared<G4Step>()}
    , update_touchable_{std::make_unique<LevelTouchableUpdater>(std::move(geo))}
{
    CELER_EXPECT(detector_volumes_ && !detector_volumes_->empty());

    CELER_LOG(debug) << "Setting up thread-local optical hit processor for "
                     << detector_volumes_->size() << " sensitive detectors";

    // Allocate secondary vector (prevents crashes in some SDs)
    step_->NewSecondaryVector();

    // Set up the single post-step point
    step_point_ = step_->GetPostStepPoint();
    CELER_ASSERT(step_point_);
    step_point_->SetStepStatus(fUserDefinedLimit);

    // Create touchable handle shared by both step points. SDs (e.g. DD4hep
    // calorimeter) may call GetPreStepPoint()->GetTouchableHandle() to compute
    // cell IDs, so the pre-step point must be valid and share the same
    // touchable as the post-step point for single-point optical hits.
    touch_handle_ = new G4TouchableHistory;
    step_point_->SetTouchableHandle(touch_handle_);
    step_->GetPreStepPoint()->SetTouchableHandle(touch_handle_);
    step_->GetPreStepPoint()->SetStepStatus(fUserDefinedLimit);

    // Mark unsupported step attributes as invalid/infinity
    step_->SetNonIonizingEnergyDeposit(
        -std::numeric_limits<double>::infinity());
    step_point_->SetLocalTime(std::numeric_limits<double>::infinity());
    step_point_->SetProperTime(std::numeric_limits<double>::infinity());
    step_point_->SetVelocity(std::numeric_limits<double>::infinity());
    step_point_->SetSafety(std::numeric_limits<double>::infinity());
    step_point_->SetPolarization(G4ThreeVector());

    // Build thread-local sensitive detector map from logical volumes
    detectors_.resize(detector_volumes_->size());
    for (auto i : range(detectors_.size()))
    {
        G4LogicalVolume const* lv = (*detector_volumes_)[i];
        CELER_ASSERT(lv);
        detectors_[i] = lv->GetSensitiveDetector();
        CELER_VALIDATE(detectors_[i],
                       << "no sensitive detector is attached to volume '"
                       << lv->GetName() << "'@"
                       << static_cast<void const*>(lv));
    }

    // Construct the single reused G4OpticalPhoton track
    G4ParticleDefinition* opticalphoton
        = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");
    CELER_VALIDATE(opticalphoton,
                   << "G4OpticalPhoton particle definition not found; "
                      "ensure optical physics is registered");

    // Create track with dummy initial position/energy; these are updated per
    // hit
    track_ = new G4Track(
        new G4DynamicParticle(opticalphoton, G4ThreeVector(0, 0, 1), 0.0),
        0.0,
        G4ThreeVector(0, 0, 0));
    track_->SetStep(step_.get());
    step_->SetTrack(track_);

    CELER_ENSURE(!detectors_.empty());
}

//---------------------------------------------------------------------------//
/*!
 * Destructor: must be called on the same thread as construction.
 */
OpticalHitProcessor::~OpticalHitProcessor()
{
    delete track_;
}

//---------------------------------------------------------------------------//
/*!
 * Process a batch of optical hits.
 *
 * For each hit, reconstruct the touchable from the volume hierarchy, populate
 * the step point, and call G4VSensitiveDetector::Hit().
 */
void OpticalHitProcessor::operator()(DetectorHitsOutput const& out)
{
    if (out.hits.empty())
        return;

    for (auto i : range(out.hits.size()))
    {
        // Get the volume instance span for this hit
        Span<VolumeInstanceId const> vol_span;
        if (out.num_volume_levels > 0)
        {
            CELER_ASSERT(out.volume_instance_ids.size()
                         == out.hits.size() * out.num_volume_levels);
            vol_span = make_span(out.volume_instance_ids)
                           .subspan(i * out.num_volume_levels,
                                    out.num_volume_levels);
        }
        this->process_hit(out.hits[i], vol_span);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process a single optical photon hit.
 */
void OpticalHitProcessor::process_hit(optical::DetectorHit const& hit,
                                      Span<VolumeInstanceId const> vol_span)
{
    CELER_EXPECT(hit);

    G4LogicalVolume const* lv = this->detector_volume(hit.detector);
    CELER_ASSERT(lv);

    // Reconstruct touchable from the full volume hierarchy
    if (!vol_span.empty())
    {
        bool success = (*update_touchable_)(vol_span, touch_handle_());
        if (CELER_UNLIKELY(!success))
        {
            CELER_LOG_LOCAL(error) << "Omitting optical hit: failed to "
                                      "reconstruct touchable";
            return;
        }
    }

    // Populate step point attributes.
    // Pre- and post-step points are set to the same position because Celeritas
    // optical hits represent a single point. DD4hep SDs compute cell IDs and
    // contribution positions from the midpoint (pre+post)/2, so pre must equal
    // post to get the correct result.
    double const g4_energy = convert_to_geant(hit.energy.value(), CLHEP::MeV);
    G4ThreeVector const g4_pos = convert_to_geant(hit.position, clhep_length);
    G4ThreeVector const g4_dir = convert_to_geant(hit.direction, 1.0);
    step_point_->SetGlobalTime(convert_to_geant(hit.time, clhep_time));
    step_point_->SetPosition(g4_pos);
    step_point_->SetMomentumDirection(g4_dir);
    step_point_->SetKineticEnergy(g4_energy);
    step_->GetPreStepPoint()->SetPosition(g4_pos);
    step_->GetPreStepPoint()->SetMomentumDirection(g4_dir);
    step_->GetPreStepPoint()->SetKineticEnergy(g4_energy);

    // Set energy deposit so calorimeter SDs record this hit
    step_->SetTotalEnergyDeposit(g4_energy);

    // Set step point material from the logical volume.
    // G4Track::GetMaterialCutsCouple() reads from the pre-step point, so set
    // it on both points so SDs that go via the track (e.g. DD4hep
    // Geant4StepHandler / G4EmSaturation) don't dereference null.
    auto* couple = lv->GetMaterialCutsCouple();
    step_point_->SetMaterial(lv->GetMaterial());
    step_point_->SetMaterialCutsCouple(couple);
    step_->GetPreStepPoint()->SetMaterial(lv->GetMaterial());
    step_->GetPreStepPoint()->SetMaterialCutsCouple(couple);
    step_point_->SetSensitiveDetector(lv->GetSensitiveDetector());

    // Update track to reflect post-step state
    CELER_ASSERT(track_);
    track_->SetGlobalTime(step_point_->GetGlobalTime());
    track_->SetPosition(step_point_->GetPosition());
    track_->SetKineticEnergy(step_point_->GetKineticEnergy());
    track_->SetMomentumDirection(step_point_->GetMomentumDirection());
    track_->SetTouchableHandle(touch_handle_);
    track_->SetNextTouchableHandle(touch_handle_);

    // Invoke the sensitive detector
    this->detector(hit.detector)->Hit(step_.get());
}

//---------------------------------------------------------------------------//
/*!
 * Build and register a thread-local OpticalHitProcessor.
 *
 * Acquires the geometry and detector volume map, constructs the processor,
 * and registers it in the thread-local registry. Returns nullptr if no
 * optical detector volumes are found.
 */
std::shared_ptr<OpticalHitProcessor>
make_optical_hit_processor(optical::CoreParams const& opt_params)
{
    auto detector_vols = build_optical_detector_volumes(opt_params);
    if (!detector_vols)
        return nullptr;

    auto geo = celeritas::global_geant_geo().lock();
    CELER_VALIDATE(geo,
                   << "GeantGeoParams required for optical hit processing");

    auto processor = std::make_shared<OpticalHitProcessor>(
        std::move(detector_vols), std::move(geo));

    thread_local_optical_hit_processor() = processor.get();

    CELER_LOG(debug) << "Built thread-local optical hit processor";
    return processor;
}

//---------------------------------------------------------------------------//
/*!
 * Unregister and destroy the thread-local OpticalHitProcessor.
 */
void reset_optical_hit_processor(std::shared_ptr<OpticalHitProcessor>& proc)
{
    if (proc)
    {
        thread_local_optical_hit_processor() = nullptr;
        proc.reset();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return a DetectorAction callback that dispatches to the thread-local
 * OpticalHitProcessor, warning when the processor is unexpectedly null.
 */
std::function<void(DetectorHitsOutput const&)> make_optical_hit_callback()
{
    return [](DetectorHitsOutput const& out) {
        auto* proc = thread_local_optical_hit_processor();
        if (proc)
        {
            (*proc)(out);
        }
        else
        {
            CELER_LOG_LOCAL(warning)
                << "Optical detector callback: "
                   "thread_local_optical_hit_processor is null, discarding "
                << out.hits.size() << " hits";
        }
    };
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
