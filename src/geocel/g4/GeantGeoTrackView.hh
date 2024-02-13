//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <type_traits>
#include <G4LogicalVolume.hh>
#include <G4Navigator.hh>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>

#include "corecel/Macros.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/Types.hh"

#include "Convert.geant.hh"
#include "GeantGeoData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Navigate through a Geant4 geometry on a single thread.
 *
 * This wraps a Geant4 geometry navigator and volume hierarchy state with the
 * same Celeritas tracker interface. It's not going to be the most efficient
 * code since the \c G4Navigator includes a lot of helper functions for
 * managing safety distance, tracking through a field, etc. We also
 * independently store a "celeritas" native position and direction, as well as
 * duplicating the "geant4" position and direction that are also stored under
 * the hood in the heavyweight navigator.
 *
 * For a description of ordering requirements, see: \sa OrangeTrackView .
 */
class GeantGeoTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = GeoTrackInitializer;
    using ParamsRef = NativeCRef<GeantGeoParamsData>;
    using StateRef = NativeRef<GeantGeoStateData>;
    using real_type = double;
    using Real3 = Array<real_type, 3>;
    //!@}

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        GeantGeoTrackView const& other;  //!< Existing geometry
        Real3 dir;  //!< New direction
    };

  public:
    // Construct from params and state data
    inline GeantGeoTrackView(ParamsRef const& params,
                             StateRef const& state,
                             TrackSlotId tid);

    // Initialize the state
    inline GeantGeoTrackView& operator=(Initializer_t const& init);
    // Initialize the state from a parent state and new direction
    inline GeantGeoTrackView& operator=(DetailedInitializer const& init);

    //// STATIC ACCESSORS ////

    //! A tiny push to make sure tracks do not get stuck at boundaries
    static constexpr real_type extra_push()
    {
        return 1e-12 * lengthunits::millimeter;
    }

    //// ACCESSORS ////

    //!@{
    //! State accessors
    CELER_FORCEINLINE Real3 const& pos() const { return pos_; }
    CELER_FORCEINLINE Real3 const& dir() const { return dir_; }
    //!@}

    // Get the volume ID in the current cell.
    CELER_FORCEINLINE VolumeId volume_id() const;
    CELER_FORCEINLINE int volume_physid() const;

    //!@{
    //! VecGeom states are never "on" a surface
    SurfaceId surface_id() const { return {}; }
    SurfaceId next_surface_id() const { return {}; }
    //!@}

    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE bool is_outside() const;
    // Whether the track is exactly on a surface
    CELER_FORCEINLINE bool is_on_boundary() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary (infinite max)
    inline Propagation find_next_step();

    // Find the distance to the next boundary, up to and including a step
    inline Propagation find_next_step(real_type max_step);

    // Find the safety at the current position
    inline real_type find_safety();

    // Find the safety at the current position up to a maximum step distance
    inline CELER_FUNCTION real_type find_safety(real_type max_step);

    // Move to the boundary in preparation for crossing it
    inline void move_to_boundary();

    // Move within the volume
    inline void move_internal(real_type step);

    // Move within the volume to a specific point
    inline void move_internal(Real3 const& pos);

    // Cross from one side of the current surface to the other
    inline void cross_boundary();

    // Change direction
    inline void set_dir(Real3 const& newdir);

  private:
    //// DATA ////

    //!@{
    //! Referenced thread-local data
    Real3& pos_;
    Real3& dir_;
    real_type& next_step_;
    real_type& safety_radius_;
    G4TouchableHandle& touch_handle_;
    G4Navigator& navi_;
    //!@}

    // Temporary data
    G4ThreeVector g4pos_;
    G4ThreeVector g4dir_;  // [mm]
    real_type g4safety_;  // [mm]

    //// HELPER FUNCTIONS ////

    // Whether any next distance-to-boundary has been found
    inline bool has_next_step() const;

    //! Get a pointer to the current volume; null if outside
    inline G4LogicalVolume const* volume() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from params and state data.
 */
GeantGeoTrackView::GeantGeoTrackView(ParamsRef const&,
                                     StateRef const& states,
                                     TrackSlotId tid)
    : pos_(states.pos[tid])
    , dir_(states.dir[tid])
    , next_step_(states.next_step[tid])
    , safety_radius_(states.safety_radius[tid])
    , touch_handle_(states.nav_state.touch_handle(tid))
    , navi_(states.nav_state.navigator(tid))
{
    g4pos_ = convert_to_geant(pos_, clhep_length);
    g4dir_ = convert_to_geant(dir_, 1);
    g4safety_ = convert_to_geant(safety_radius_, clhep_length);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 */
GeantGeoTrackView& GeantGeoTrackView::operator=(Initializer_t const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    // Initialize position/direction
    std::copy(init.pos.begin(), init.pos.end(), pos_.begin());
    std::copy(init.dir.begin(), init.dir.end(), dir_.begin());
    next_step_ = 0;
    safety_radius_ = -1;  // Assume *not* on a boundary

    g4pos_ = convert_to_geant(pos_, clhep_length);
    g4dir_ = convert_to_geant(dir_, 1);
    g4safety_ = -1;

    navi_.LocateGlobalPointAndUpdateTouchable(g4pos_,
                                              g4dir_,
                                              touch_handle_(),
                                              /* relative_search = */ false);

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state from a direction and a copy of the parent state.
 *
 * \c G4Track::SetTouchableHandle from \c G4VEmProcess::PostStepDoIt
 *
 * maybe see \c G4SteppingManager::Stepping
 */
GeantGeoTrackView& GeantGeoTrackView::operator=(DetailedInitializer const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    if (this != &init.other)
    {
        // Copy values from the parent state
        pos_ = init.other.pos_;
        safety_radius_ = init.other.safety_radius_;
        g4pos_ = init.other.g4pos_;
        g4dir_ = init.other.g4dir_;
        g4safety_ = init.other.g4safety_;

        // Update the touchable and navigator
        touch_handle_ = init.other.touch_handle_;
        navi_.ResetHierarchyAndLocate(
            g4pos_, g4dir_, dynamic_cast<G4TouchableHistory&>(*touch_handle_()));
    }

    // Set up the next state and initialize the direction
    dir_ = init.dir;
    next_step_ = 0;

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
VolumeId GeantGeoTrackView::volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    auto inst_id = this->volume()->GetInstanceID();
    CELER_ENSURE(inst_id >= 0);
    return VolumeId{static_cast<VolumeId::size_type>(inst_id)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the physical volume ID in the current cell.
 */
int GeantGeoTrackView::volume_physid() const
{
    CELER_EXPECT(!this->is_outside());
    G4VPhysicalVolume* pv = touch_handle_()->GetVolume(0);
    if (!pv)
        return -1;
    return pv->GetInstanceID();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
bool GeantGeoTrackView::is_outside() const
{
    return this->volume() == nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is on the boundary of a volume.
 */
bool GeantGeoTrackView::is_on_boundary() const
{
    return safety_radius_ == 0.0;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
Propagation GeantGeoTrackView::find_next_step()
{
    return this->find_next_step(numeric_limits<real_type>::infinity());
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 *
 * It seems that ComputeStep cannot be called twice in a row without an
 * intermediate call to \c LocateGlobalPointWithinVolume: the safety will be
 * set to zero.
 */
Propagation GeantGeoTrackView::find_next_step(real_type max_step)
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(max_step > 0);

    // Compute the step
    real_type g4step = convert_to_geant(max_step, clhep_length);
    g4step = navi_.ComputeStep(g4pos_, g4dir_, g4step, g4safety_);

    if (g4safety_ != 0 && !this->is_on_boundary())
    {
        // Save the resulting safety distance if computed: allow to be
        // "negative" to prevent accidentally changing the boundary state
        safety_radius_ = convert_from_geant(g4safety_, clhep_length);
        CELER_ASSERT(!this->is_on_boundary());
    }

    // Update result
    Propagation result;
    result.distance = convert_from_geant(g4step, clhep_length);
    if (result.distance <= max_step)
    {
        result.boundary = true;
        result.distance
            = celeritas::max<real_type>(result.distance, this->extra_push());
        CELER_ENSURE(result.distance > 0);
    }
    else
    {
        // No intersection in range -> G4Navigator returns kInfinity
        result.distance = max_step;
        CELER_ENSURE(result.distance > 0);
    }

    // Save the next step
    next_step_ = result.distance;

    CELER_ENSURE(result.distance > 0);
    CELER_ENSURE(result.distance <= max(max_step, this->extra_push()));
    CELER_ENSURE(result.boundary || result.distance == max_step
                 || max_step < this->extra_push());
    CELER_ENSURE(this->has_next_step());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety at the current position.
 */
auto GeantGeoTrackView::find_safety() -> real_type
{
    return this->find_safety(numeric_limits<real_type>::infinity());
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety at the current position.
 *
 * \warning This can change the boundary state if the track was moved to or
 * initialized a point on the boundary.
 */
auto GeantGeoTrackView::find_safety(real_type max_step) -> real_type
{
    CELER_EXPECT(max_step > 0);
    if (!this->is_on_boundary() && (safety_radius_ < max_step))
    {
        real_type g4step = convert_to_geant(max_step, clhep_length);
        g4safety_ = navi_.ComputeSafety(g4pos_, g4step);
        safety_radius_ = max(convert_from_geant(g4safety_, clhep_length), 0.0);
    }

    return safety_radius_;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary but don't cross yet.
 */
void GeantGeoTrackView::move_to_boundary()
{
    CELER_EXPECT(this->has_next_step());

    // Move next step
    axpy(next_step_, dir_, &pos_);
    axpy(convert_to_geant(next_step_, clhep_length), g4dir_, &g4pos_);
    next_step_ = 0;
    safety_radius_ = 0;
    g4safety_ = 0;
    navi_.SetGeometricallyLimitedStep();

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary.
 */
void GeantGeoTrackView::cross_boundary()
{
    CELER_EXPECT(this->is_on_boundary());

    navi_.LocateGlobalPointAndUpdateTouchableHandle(
        g4pos_,
        g4dir_,
        touch_handle_,
        /* relative_search = */ true);

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume.
 *
 * The straight-line distance *must* be less than the distance to the
 * boundary.
 */
void GeantGeoTrackView::move_internal(real_type dist)
{
    CELER_EXPECT(this->has_next_step());
    CELER_EXPECT(dist > 0 && dist <= next_step_);

    // Move and update next_step
    axpy(dist, dir_, &pos_);
    axpy(convert_to_geant(dist, clhep_length), g4dir_, &g4pos_);
    next_step_ -= dist;
    navi_.LocateGlobalPointWithinVolume(g4pos_);

    safety_radius_ = -1;
    g4safety_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume to a nearby point.
 *
 * See \c G4PathFinder::ReLocate from \c G4SafetyHelper::ReLocateWithinVolume
 * from \c G4VMultipleScattering::AlongStepDoIt .
 */
void GeantGeoTrackView::move_internal(Real3 const& pos)
{
    pos_ = pos;
    g4pos_ = convert_to_geant(pos_, clhep_length);
    next_step_ = 0;
    navi_.LocateGlobalPointWithinVolume(g4pos_);

    safety_radius_ = -1;
    g4safety_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Change the track's direction.
 *
 * This happens after a scattering event or movement inside a magnetic field.
 * It resets the calculated distance-to-boundary.
 */
void GeantGeoTrackView::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));
    dir_ = newdir;
    g4dir_ = convert_to_geant(newdir, 1);
    next_step_ = 0;
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether a next step has been calculated.
 */
bool GeantGeoTrackView::has_next_step() const
{
    return next_step_ != 0;
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
 */
auto GeantGeoTrackView::volume() const -> G4LogicalVolume const*
{
    CELER_EXPECT(touch_handle_());
    G4VPhysicalVolume* pv = touch_handle_()->GetVolume(0);
    if (!pv)
        return nullptr;

    return pv->GetLogicalVolume();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
