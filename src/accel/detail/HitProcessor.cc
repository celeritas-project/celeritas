//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.cc
//---------------------------------------------------------------------------//
#include "HitProcessor.hh"

#include <string>
#include <utility>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>
#include <G4Navigator.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHistory.hh>
#include <G4Track.hh>
#include <G4TransportationManager.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VSensitiveDetector.hh>
#include <G4Version.hh>

#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/Convert.geant.hh"
#include "celeritas/ext/GeantGeoUtils.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<>
struct ReprTraits<G4ThreeVector>
{
    using value_type = std::decay_t<G4ThreeVector>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "G4ThreeVector";
        if (name)
        {
            os << ' ' << name;
        }
    }
    static void init(std::ostream& os) { ReprTraits<double>::init(os); }

    static void print_value(std::ostream& os, G4ThreeVector const& vec)
    {
        os << '{' << vec[0] << ", " << vec[1] << ", " << vec[2] << '}';
    }
};

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct local navigator and step data.
 */
HitProcessor::HitProcessor(SPConstVecLV detector_volumes,
                           VecParticle const& particles,
                           StepSelection const& selection,
                           bool locate_touchable)
    : detector_volumes_(std::move(detector_volumes))
{
    CELER_EXPECT(detector_volumes_ && !detector_volumes_->empty());
    CELER_VALIDATE(!locate_touchable || selection.points[StepPoint::pre].pos,
                   << "cannot set 'locate_touchable' because the pre-step "
                      "position is not being collected");
    CELER_VALIDATE(!locate_touchable || selection.points[StepPoint::pre].pos,
                   << "cannot set 'locate_touchable' because the pre-step "
                      "position is not being collected");

    // Create temporary objects
    step_ = std::make_unique<G4Step>();

#if G4VERSION_NUMBER >= 1103
#    define HP_CLEAR_STEP_POINT(CMD) step_->CMD(nullptr)
#else
#    define HP_CLEAR_STEP_POINT(CMD) /* no "reset" before v11.0.3 */
#endif

#define HP_SETUP_POINT(LOWER, TITLE)                                          \
    do                                                                        \
    {                                                                         \
        if (!selection.points[StepPoint::LOWER])                              \
        {                                                                     \
            HP_CLEAR_STEP_POINT(Reset##TITLE##StepPoint);                     \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            step_->Get##TITLE##StepPoint()->SetStepStatus(fUserDefinedLimit); \
        }                                                                     \
    } while (0)

    HP_SETUP_POINT(pre, Pre);
    HP_SETUP_POINT(post, Post);
#undef HP_SETUP_POINT
#undef HP_CLEAR_STEP_POINT

    // Create navigator
    G4VPhysicalVolume* world_volume
        = G4TransportationManager::GetTransportationManager()
              ->GetNavigatorForTracking()
              ->GetWorldVolume();
    if (locate_touchable)
    {
        CELER_ASSERT(selection.points[StepPoint::pre].pos
                     && selection.points[StepPoint::pre].dir);
        navi_ = std::make_unique<G4Navigator>();
        navi_->SetWorldVolume(world_volume);

        // Create "touchable handle" (shared pointer to G4TouchableHistory)
        touch_handle_ = new G4TouchableHistory;
        step_->GetPreStepPoint()->SetTouchableHandle(touch_handle_);
    }

    // Create track
    for (G4ParticleDefinition const* pd : particles)
    {
        CELER_ASSERT(pd);
        tracks_.emplace_back(new G4Track(
            new G4DynamicParticle(pd, G4ThreeVector()), 0.0, G4ThreeVector()));
        tracks_.back()->SetTrackID(-1);
        tracks_.back()->SetParentID(-1);
    }

    // Convert logical volumes (global) to sensitive detectors (thread local)
    CELER_LOG_LOCAL(debug) << "Setting up " << detector_volumes_->size()
                           << " sensitive detectors";
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

    CELER_ENSURE(!detectors_.empty());
}

//---------------------------------------------------------------------------//
HitProcessor::~HitProcessor() = default;

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (CPU).
 */
void HitProcessor::operator()(StepStateHostRef const& states)
{
    copy_steps(&steps_, states);
    if (steps_)
    {
        (*this)(steps_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void HitProcessor::operator()(StepStateDeviceRef const& states)
{
    copy_steps(&steps_, states);
    if (steps_)
    {
        (*this)(steps_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Generate and call hits from a detector output.
 *
 * In an application setting, this is always called with our local data \c
 * steps_ as an argument. For tests, we can call this function explicitly using
 * local test data.
 */
void HitProcessor::operator()(DetectorStepOutput const& out) const
{
    CELER_EXPECT(!out.detector.empty());
    CELER_ASSERT(!navi_ || !out.points[StepPoint::pre].pos.empty());
    CELER_ASSERT(!navi_ || !out.points[StepPoint::pre].dir.empty());
    CELER_ASSERT(tracks_.empty() || !out.particle.empty());

    CELER_LOG_LOCAL(debug) << "Processing " << out.size() << " hits";

    EnumArray<StepPoint, G4StepPoint*> points
        = {step_->GetPreStepPoint(), step_->GetPostStepPoint()};

    for (auto i : range(out.size()))
    {
#define HP_SET(SETTER, OUT, UNITS)                   \
    do                                               \
    {                                                \
        if (!OUT.empty())                            \
        {                                            \
            SETTER(convert_to_geant(OUT[i], UNITS)); \
        }                                            \
    } while (0)

        HP_SET(step_->SetTotalEnergyDeposit, out.energy_deposition, CLHEP::MeV);

        for (auto sp : range(StepPoint::size_))
        {
            if (!points[sp])
            {
                continue;
            }
            HP_SET(points[sp]->SetGlobalTime, out.points[sp].time, CLHEP::s);
            HP_SET(points[sp]->SetPosition, out.points[sp].pos, CLHEP::cm);
            HP_SET(points[sp]->SetKineticEnergy,
                   out.points[sp].energy,
                   CLHEP::MeV);
            HP_SET(points[sp]->SetMomentumDirection, out.points[sp].dir, 1);
        }
#undef HP_SET

        if (navi_)
        {
            G4LogicalVolume const* lv = this->detector_volume(out.detector[i]);

            // Update navigation state
            constexpr auto sp = StepPoint::pre;
            bool success = this->update_touchable(out.points[sp].pos[i],
                                                  out.points[sp].dir[i],
                                                  lv,
                                                  touch_handle_());
            if (CELER_UNLIKELY(!success))
            {
                // Inconsistent touchable: skip this energy deposition
                continue;
            }

            // Copy attributes from logical volume
            points[sp]->SetMaterial(lv->GetMaterial());
            points[sp]->SetMaterialCutsCouple(lv->GetMaterialCutsCouple());
        }

        if (!tracks_.empty())
        {
            this->update_track(out.particle[i]);
        }

        // Hit sensitive detector
        this->detector(out.detector[i])->Hit(step_.get());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Recreate the track from the particle ID and saved post-step data.
 */
void HitProcessor::update_track(ParticleId id) const
{
    CELER_EXPECT(id < tracks_.size());
    G4Track& track = *tracks_[id.unchecked_get()];
    step_->SetTrack(&track);

    if (G4StepPoint* pre_step = step_->GetPreStepPoint())
    {
        // Copy data from track to pre-step
        G4ParticleDefinition const& pd = *track.GetParticleDefinition();
        pre_step->SetCharge(pd.GetPDGCharge());
    }
    if (G4StepPoint* post_step = step_->GetPostStepPoint())
    {
        // Copy data from post-step to track
        track.SetGlobalTime(post_step->GetGlobalTime());
        track.SetPosition(post_step->GetPosition());
        track.SetKineticEnergy(post_step->GetKineticEnergy());
        track.SetMomentumDirection(post_step->GetMomentumDirection());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Update the temporary navigation state based on the position and direction.
 */
bool HitProcessor::update_touchable(Real3 const& pos,
                                    Real3 const& dir,
                                    G4LogicalVolume const* lv,
                                    G4VTouchable* touchable) const
{
    auto g4pos = convert_to_geant(pos, CLHEP::cm);
    auto g4dir = convert_to_geant(dir, 1);

    // Locate pre-step point
    navi_->LocateGlobalPointAndUpdateTouchable(g4pos,
                                               g4dir,
                                               touchable,
                                               /* relative_search = */ false);

    // Check that physical and logical volumes are consistent
    G4VPhysicalVolume* pv = touchable->GetVolume(0);
    CELER_ASSERT(pv);
    if (!CELER_UNLIKELY((pv->GetLogicalVolume() != lv)))
    {
        return true;
    }

    // We may be accidentally in the old volume and crossing into
    // the new one: try crossing the edge. Use a fairly loose tolerance since
    // there may be small differences between the Geant4 and VecGeom
    // representations of the geometry.
    double safety_distance{-1};
    constexpr double max_step = 1 * CLHEP::mm;
    double step = navi_->ComputeStep(g4pos, g4dir, max_step, safety_distance);
    if (step < max_step)
    {
        // Found a nearby volume
        if (step > 1e-3 * CLHEP::mm)
        {
            // Warn only if the step is nontrivial
            CELER_LOG(warning)
                << "Bumping navigation state by " << repr(step / CLHEP::mm)
                << " [mm] because the pre-step point at " << repr(g4pos)
                << " [mm] along " << repr(dir)
                << " is expected to be in logical volume " << PrintableLV{lv}
                << " but navigation gives " << PrintableNavHistory{touchable};
        }

        navi_->SetGeometricallyLimitedStep();
        navi_->LocateGlobalPointAndUpdateTouchable(
            g4pos,
            g4dir,
            touchable,
            /* relative_search = */ true);

        pv = touchable->GetVolume(0);
        CELER_ASSERT(pv);
    }
    else
    {
        // No nearby crossing found
        CELER_LOG(warning)
            << "Failed to bump navigation state up to a distance of "
            << max_step / CLHEP::mm << " [mm]";
    }

    if (CELER_UNLIKELY(pv->GetLogicalVolume() != lv))
    {
        CELER_LOG(error)
            << "expected step point at " << repr(g4pos) << " [mm] along "
            << repr(dir) << " to be in logical volume " << PrintableLV{lv}
            << " but navigation gives " << PrintableNavHistory{touchable}
            << ": omitting energy deposition of "
            << step_->GetTotalEnergyDeposit() / CLHEP::MeV << " [MeV]";
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
