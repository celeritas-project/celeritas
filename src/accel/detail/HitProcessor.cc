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
namespace
{
//---------------------------------------------------------------------------//
template<class T>
inline T convert_to_geant(T const& val, T units)
{
    return val * units;
}

//---------------------------------------------------------------------------//
inline G4ThreeVector convert_to_geant(Real3 const& arr, double units)
{
    return {arr[0] * units, arr[1] * units, arr[2] * units};
}

//---------------------------------------------------------------------------//
inline double convert_to_geant(units::MevEnergy const& energy, double units)
{
    CELER_EXPECT(units == CLHEP::MeV);
    return energy.value() * CLHEP::MeV;
}

//---------------------------------------------------------------------------//
struct PrintableNavHistory
{
    G4VTouchable* touch{nullptr};
};

std::ostream& operator<<(std::ostream& os, PrintableNavHistory const& pnh)
{
    CELER_EXPECT(pnh.touch);
    os << '{';

    G4VTouchable& touch = *pnh.touch;
    for (int depth : range(touch.GetHistoryDepth()))
    {
        G4VPhysicalVolume* vol = touch.GetVolume(depth);
        CELER_ASSERT(vol);
        G4LogicalVolume* lv = vol->GetLogicalVolume();
        CELER_ASSERT(lv);
        if (depth != 0)
        {
            os << " -> ";
        }
        os << "{pv='" << vol->GetName() << "', lv=" << lv->GetInstanceID()
           << "='" << lv->GetName() << "'}";
    }
    os << '}';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct local navigator and step data.
 */
HitProcessor::HitProcessor(VecLV detector_volumes,
                           StepSelection const& selection,
                           bool locate_touchable)
    : detector_volumes_(std::move(detector_volumes))
{
    CELER_EXPECT(!detector_volumes_.empty());
    CELER_VALIDATE(!locate_touchable || selection.points[StepPoint::pre].pos,
                   << "cannot set 'locate_touchable' because the pre-step "
                      "position is not being collected");

    // Create temporary objects
    step_ = std::make_unique<G4Step>();

#if G4VERSION_NUMBER >= 1101
#    define HP_CLEAR_STEP_POINT(CMD) step_->CMD(nullptr)
#else
#    define HP_CLEAR_STEP_POINT(CMD) /* no "reset" before v11.0.1 */
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
    if (locate_touchable && selection.points[StepPoint::pre].pos)
    {
        navi_ = std::make_unique<G4Navigator>();
        navi_->SetWorldVolume(world_volume);

        // Create "touchable handle" (shared pointer to G4TouchableHistory)
        touch_handle_ = new G4TouchableHistory;
        step_->GetPreStepPoint()->SetTouchableHandle(touch_handle_);
    }
}

//---------------------------------------------------------------------------//
HitProcessor::~HitProcessor() = default;

//---------------------------------------------------------------------------//
/*!
 * Generate and call hits from a detector output.
 */
void HitProcessor::operator()(DetectorStepOutput const& out) const
{
    CELER_EXPECT(!out.detector.empty());
    CELER_ASSERT(!navi_ || !out.points[StepPoint::pre].pos.empty());
    CELER_ASSERT(!navi_ || !out.points[StepPoint::pre].dir.empty());

    CELER_LOG_LOCAL(debug) << "Processing " << out.size() << " hits";

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
        // TODO: how to handle these attributes?
        // step_->SetTrack(primary_track);

        // TODO: assert that event ID is consistent with active
        // LocalTransporter event?

        EnumArray<StepPoint, G4StepPoint*> points
            = {step_->GetPreStepPoint(), step_->GetPostStepPoint()};
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
            // TODO: do we set secondary properties like the velocity,
            // material, mass, charge,  ... ?

            // TODO: how to handle these attributes?
            // step_->SetTrack(primary_track);
            // dynamic particle, ParticleDefinition
            // pre->SetLocalTime
            // pre->SetProperTime
            // pre->SetTouchableHandle
        }
#undef HP_SET

        if (navi_)
        {
            CELER_ASSERT(out.detector[i] < detector_volumes_.size());
            bool success = this->update_touchable(
                out.points[StepPoint::pre].pos[i],
                out.points[StepPoint::pre].dir[i],
                detector_volumes_[out.detector[i].unchecked_get()]);
            if (CELER_UNLIKELY(!success))
            {
                // Inconsistent touchable: skip this energy deposition
                continue;
            }
        }

        // Hit sensitive detector (NOTE: GetSensitiveDetector returns a
        // thread-local object from a global object.)
        CELER_ASSERT(out.detector[i] < detector_volumes_.size());
        G4VSensitiveDetector* sd
            = detector_volumes_[out.detector[i].unchecked_get()]
                  ->GetSensitiveDetector();
        CELER_ASSERT(sd);
        sd->Hit(step_.get());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Update the temporary navigation state based on the position and direction.
 */
bool HitProcessor::update_touchable(Real3 const& pos,
                                    Real3 const& dir,
                                    G4LogicalVolume* lv) const
{
    auto g4pos = convert_to_geant(pos, CLHEP::cm);
    auto g4dir = convert_to_geant(dir, 1);
    G4VTouchable* touchable = touch_handle_();

    // Locate pre-step point
    navi_->LocateGlobalPointAndUpdateTouchable(g4pos,
                                               g4dir,
                                               touchable,
                                               /* relative_search = */ false);

    // Check that physical and logical volumes are consistent
    G4VPhysicalVolume* pv = touchable->GetVolume(0);
    CELER_ASSERT(pv);
    if (!CELER_UNLIKELY(pv->GetLogicalVolume() != lv))
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
        if (step > 0)
        {
            // Warn only if the step is nontrivial
            CELER_LOG(warning)
                << "Bumping navigation state by " << repr(step / CLHEP::mm)
                << " [mm] because the pre-step point at " << repr(g4pos)
                << " [mm] along " << repr(dir)
                << " is expected to be in logical volume '" << lv->GetName()
                << "' (ID " << lv->GetInstanceID() << ") but navigation gives "
                << PrintableNavHistory{touchable};
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
            << repr(dir) << " to be in logical volume '" << lv->GetName()
            << "' (ID " << lv->GetInstanceID() << ") but navigation gives "
            << PrintableNavHistory{touchable}
            << ": omitting energy deposition of "
            << step_->GetTotalEnergyDeposit() / CLHEP::MeV << " [MeV]";
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
