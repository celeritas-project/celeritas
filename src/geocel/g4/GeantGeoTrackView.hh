//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoTrackView.hh
//! \sa geocel/g4/GeantGeo.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <G4LogicalVolume.hh>
#include <G4Navigator.hh>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>
#include <G4VPhysicalVolume.hh>

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/Types.hh"
#include "geocel/detail/GeantVolumeInstanceMapper.hh"

#include "Convert.hh"
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
 * \internal
 *
 * Geant4 thinks about boundaries differently to Celeritas. When we find the
 * next step, it will internally store the "blocking" volume
 * (daughter="entering", current volume="exiting") and the normal at that
 * point. The G4 navigation in practice has three options:
 *
 * 1. move within the safety distance, which is calculated as a side effect of
 *  \c ComputeStep
 * 2. move to the end-of-step position, call \c SetGeometricallyLimitedStep,
 *   and update the volume via \c LocateGlobalPointAndUpdateTouchableHandle
 * 3. move to a position outside the safety and update with \c
 *   LocateGlobalPointWithinVolume .
 *
 * \sa GeoTrackInterface
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

  public:
    // Construct from params and state data
    inline GeantGeoTrackView(
        ParamsRef const& params, StateRef const& state, TrackSlotId tid);

    // Initialize the state
    inline GeantGeoTrackView& operator=(Initializer_t const& init);

    //// ACCESSORS ////

    //!@{
    //! State accessors
    CELER_FORCEINLINE Real3 const& pos() const { return pos_; }
    CELER_FORCEINLINE Real3 const& dir() const { return dir_; }
    //!@}

    // Get the canonical volume ID in the current impl volume
    inline VolumeId volume_id() const;
    // Get the physical volume ID in the current cell
    inline VolumeInstanceId volume_instance_id() const;
    // Get the depth in the geometry hierarchy
    inline VolumeLevelId volume_level() const;
    // Get the volume instance ID for all levels
    inline void volume_instance_id(Span<VolumeInstanceId> levels) const;
    // Visit every volume instance in the track's path, including world
    template<class F>
    inline void foreach_volume_path(F&& visit) const;

    // Get the implementation volume ID
    inline ImplVolumeId impl_volume_id() const;

    // Geometry tracking status
    inline GeoStatus geo_status() const;
    // Whether the track is outside the valid geometry region
    inline bool is_outside() const;
    // Whether the track is exactly on a surface
    inline bool is_on_boundary() const;
    // Whether the last operation resulted in an error
    inline bool failed() const;
    // Get the normal vector of the current surface
    inline Real3 const& normal() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary, up to and including a step
    inline Propagation find_next_step(real_type max_step);

    // Find the safety at the current position
    inline real_type find_safety();

    // Find the safety at the current position up to a maximum step distance
    inline real_type find_safety(real_type max_step);

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

    //// IMPLEMENTATION ////

    // Get the Geant4 navigation state
    inline G4NavigationHistory const* nav_history() const;

  private:
    //// TYPES ////

    //! Helper struct for initializing from an existing geometry state
    struct DetailedInitializer
    {
        TrackSlotId parent;  //!< Parent track with existing geometry
        ::celeritas::Real3 const& dir;  //!< New direction
    };
    using ClhepLength = lengthunits::ClhepLength;

    //// DATA ////

    // Shared data
    ParamsRef const& params_;
    // Geometry state data
    //! \todo This is only needed for the detailed initialization
    StateRef const& state_;
    TrackSlotId tid_;

    //!@{
    //! Referenced thread-local data
    Real3& pos_;
    Real3& dir_;
    Real3& normal_;
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

    // Initialize the state from a parent state and new direction
    inline GeantGeoTrackView& operator=(DetailedInitializer const& init);

    // Whether any next distance-to-boundary has been found
    inline bool has_next_step() const;

    // Whether the track direction is exiting the current volume
    inline bool is_dir_exiting() const;

    // Set the geometry tracking status
    inline void geo_status(GeoStatus);

    //! Get a pointer to the current volume; null if outside
    inline G4LogicalVolume const* volume() const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from params and state data.
 */
GeantGeoTrackView::GeantGeoTrackView(
    ParamsRef const& params, StateRef const& states, TrackSlotId tid)
    : params_{params}
    , state_(states)
    , tid_(tid)
    , pos_(states.pos[tid])
    , dir_(states.dir[tid])
    , normal_(states.normal[tid])
    , next_step_(states.next_step[tid])
    , safety_radius_(states.safety_radius[tid])
    , touch_handle_(states.nav_state.touch_handle(tid))
    , navi_(states.nav_state.navigator(tid))
    , g4pos_(native_to_geant<ClhepLength>(pos_))
    , g4dir_(to_g4vector(dir_))
    , g4safety_(native_to_geant<ClhepLength>(safety_radius_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 */
GeantGeoTrackView& GeantGeoTrackView::operator=(Initializer_t const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));

    if (init.parent)
    {
        // Initialize from direction and copy of parent state
        *this = DetailedInitializer{init.parent, init.dir};
        return *this;
    }

    // Initialize position/direction
    std::copy(init.pos.begin(), init.pos.end(), pos_.begin());
    std::copy(init.dir.begin(), init.dir.end(), dir_.begin());
    next_step_ = 0;
    safety_radius_ = -1;  // Assume *not* on a boundary

    g4pos_ = native_to_geant<ClhepLength>(pos_);
    g4dir_ = to_g4vector(dir_);
    g4safety_ = -1;

    navi_.LocateGlobalPointAndUpdateTouchable(g4pos_,
                                              g4dir_,
                                              touch_handle_(),
                                              /* relative_search = */ false);
    this->geo_status(
        this->is_outside() ? GeoStatus::invalid : GeoStatus::interior);

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

    if (tid_ != init.parent)
    {
        // Copy values from the parent state
        GeantGeoTrackView other(ParamsRef{}, state_, init.parent);
        pos_ = other.pos_;
        safety_radius_ = other.safety_radius_;
        g4pos_ = other.g4pos_;
        g4dir_ = other.g4dir_;
        g4safety_ = other.g4safety_;

        // Update the touchable and navigator
        touch_handle_ = other.touch_handle_;
        navi_.ResetHierarchyAndLocate(
            g4pos_, g4dir_, dynamic_cast<G4TouchableHistory&>(*touch_handle_()));
        this->geo_status(state_.status[init.parent]);
    }

    // Set up the next state and initialize the direction
    std::copy(init.dir.begin(), init.dir.end(), dir_.begin());
    next_step_ = 0;

    CELER_ENSURE(!this->has_next_step());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the canonical volume ID in the current implementation volume.
 *
 * \note See GeantGeoParams::volume_id : for now, identical to ImplVolumeId
 */
VolumeId GeantGeoTrackView::volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return id_cast<VolumeId>(this->impl_volume_id().get());
}

//---------------------------------------------------------------------------//
/*!
 * Get the physical volume ID in the current cell.
 */
VolumeInstanceId GeantGeoTrackView::volume_instance_id() const
{
    CELER_EXPECT(!this->is_outside());
    G4VPhysicalVolume* pv = touch_handle_()->GetVolume(0);
    CELER_ASSERT(pv);
    return params_.vi_mapper->geant_to_id(*pv);
}

//---------------------------------------------------------------------------//
/*!
 * Get the depth in the geometry hierarchy.
 */
VolumeLevelId GeantGeoTrackView::volume_level() const
{
    auto* touch = touch_handle_();
    return id_cast<VolumeLevelId>(touch->GetHistoryDepth());
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume instance ID at every level.
 *
 * The input span size must be equal to the value of "volume_level" plus one.
 * The top-most level ("world" or level zero) starts at index zero and moves
 * downward. Note that Geant4 uses the \em reverse nomenclature.
 */
void GeantGeoTrackView::volume_instance_id(Span<VolumeInstanceId> levels) const
{
    this->foreach_volume_path(
        [levels](VolumeLevelId lev, VolumeInstanceId vol_inst) {
            CELER_EXPECT(lev < levels.size());
            CELER_EXPECT(vol_inst);
            levels[*lev] = vol_inst;
        });
}

//---------------------------------------------------------------------------//
/*!
 * Apply the function with the volume instance ID and level.
 *
 * This can be used to construct a unique volume instance ID or fill a vector
 * with volume levels. It is performed in local-to-global order.
 */
template<class F>
void GeantGeoTrackView::foreach_volume_path(F&& visit) const
{
    auto* touch = touch_handle_();
    auto const num_vol_levels
        = id_cast<VolumeLevelId>(touch->GetHistoryDepth());
    for (auto lev : range(num_vol_levels + 1))
    {
        VolumeInstanceId vi_id;
        G4VPhysicalVolume* pv = touch->GetVolume(num_vol_levels - lev);
        CELER_ASSERT(pv);
        vi_id = params_.vi_mapper->geant_to_id(*pv);
        visit(lev, vi_id);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
ImplVolumeId GeantGeoTrackView::impl_volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return id_cast<ImplVolumeId>(
        this->volume()->GetInstanceID() - params_.lv_offset);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FORCEINLINE bool GeantGeoTrackView::is_outside() const
{
    return this->volume() == nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is on the boundary of a volume.
 */
CELER_FORCEINLINE bool GeantGeoTrackView::is_on_boundary() const
{
    return safety_radius_ == 0.0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the last operation resulted in an error.
 */
CELER_FORCEINLINE_FUNCTION bool GeantGeoTrackView::failed() const
{
    return this->geo_status() == GeoStatus::error;
}

//---------------------------------------------------------------------------//
/*!
 * Geometry tracking status.
 */
GeoStatus GeantGeoTrackView::geo_status() const
{
    return state_.status[tid_];
}

//---------------------------------------------------------------------------//
/*!
 * Get the exit surface normal of the boundary the track has just crossed.
 *
 * This vector is in the global coordinate system.
 */
auto GeantGeoTrackView::normal() const -> Real3 const&
{
    CELER_EXPECT(this->is_on_boundary());
    return normal_;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
Propagation GeantGeoTrackView::find_next_step(real_type max_step)
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(max_step > 0);

    if (this->geo_status() == GeoStatus::boundary_inc)
    {
        // On a boundary, headed in: next step is zero
        return {0, true};
    }

    if (this->geo_status() == GeoStatus::boundary_out)
    {
        if (!this->is_dir_exiting())
        {
            // We moved to a boundary but ended up not crossing it. Tell the
            // navigator to "relocate" to avoid a warning: the current
            // position is likely outside the last calculated safety, but we
            // know it's correct because we did it via a straight-line distance
            // that's valid.
            navi_.LocateGlobalPointWithinVolume(g4pos_);
        }
    }

    // Compute the step
    real_type g4step = native_to_geant<ClhepLength>(max_step);
    g4step = navi_.ComputeStep(g4pos_, g4dir_, g4step, g4safety_);

    if (g4safety_ != 0 && !this->is_on_boundary())
    {
        // Save the resulting safety distance if computed: allow to be
        // "negative" to prevent accidentally changing the boundary state
        safety_radius_ = native_value_from(ClhepLength{g4safety_});
        CELER_ASSERT(!this->is_on_boundary());
    }

    // Update result
    Propagation result;
    result.distance = native_value_from(ClhepLength{g4step});
    if (result.distance <= max_step)
    {
        if (CELER_UNLIKELY(result.distance == 0))
        {
            CELER_LOG_LOCAL(warning) << "Boundary direction was inconsistent";
            // Oops, we might be on the inside of a concave object (boundary
            // direction was incorrect)
            this->geo_status(GeoStatus::boundary_inc);
            // On a boundary, headed in: next step is zero
            return {0, true};
        }
        result.boundary = true;
    }
    else
    {
        // No intersection in range -> G4Navigator returns kInfinity
        result.distance = max_step;
    }

    // Save the next step
    next_step_ = result.distance;

    CELER_ENSURE(result.distance > 0);
    CELER_ENSURE(result.distance <= max_step);
    CELER_ENSURE(result.boundary || result.distance == max_step);
    CELER_ENSURE(this->has_next_step());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety at the current position.
 */
CELER_FORCEINLINE auto GeantGeoTrackView::find_safety() -> real_type
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
    CELER_EXPECT(!this->is_on_boundary());
    CELER_EXPECT(max_step > 0);
    if (safety_radius_ < max_step)
    {
        real_type g4step = native_to_geant<ClhepLength>(max_step);
        g4safety_ = navi_.ComputeSafety(g4pos_, g4step);
        safety_radius_ = max(native_value_from(ClhepLength{g4safety_}), 0.0);
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
    axpy(native_to_geant<ClhepLength>(next_step_), g4dir_, &g4pos_);
    next_step_ = 0;
    safety_radius_ = 0;
    g4safety_ = 0;

    // Cache the exit normal (valid immediately after ComputeStep hits a
    // boundary)
    bool normal_valid{false};
    G4ThreeVector g4norm = navi_.GetGlobalExitNormal(g4pos_, &normal_valid);
    CELER_ASSERT(normal_valid);
    normal_ = to_array(g4norm);
    CELER_ASSERT(this->is_dir_exiting());
    this->geo_status(GeoStatus::boundary_inc);

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position \em must be on the boundary following a move-to-boundary.
 */
void GeantGeoTrackView::cross_boundary()
{
    CELER_EXPECT(this->is_on_boundary());

    if (this->geo_status() == GeoStatus::boundary_out)
    {
        // Direction changed while on boundary leading to no change in
        // volume/surface. This is logically equivalent to a reflection.
        return;
    }
    if (!this->is_dir_exiting())
    {
        // Reentering after leaving volume but not moving: we have to search
        // for the boundary again
        navi_.ComputeStep(g4pos_, g4dir_, 1e-9, g4safety_);
    }

    navi_.SetGeometricallyLimitedStep();
    navi_.LocateGlobalPointAndUpdateTouchableHandle(
        g4pos_,
        g4dir_,
        touch_handle_,
        /* relative_search = */ true);

    this->geo_status(GeoStatus::boundary_out);

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
    axpy(native_to_geant<ClhepLength>(dist), g4dir_, &g4pos_);
    next_step_ -= dist;
    navi_.LocateGlobalPointWithinVolume(g4pos_);

    safety_radius_ = -1;
    g4safety_ = 0;
    this->geo_status(GeoStatus::interior);
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
    g4pos_ = native_to_geant<ClhepLength>(pos_);
    next_step_ = 0;
    navi_.LocateGlobalPointWithinVolume(g4pos_);

    safety_radius_ = -1;
    g4safety_ = 0;
    this->geo_status(GeoStatus::interior);
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

    GeoStatus status{this->geo_status()};
    if (::celeritas::is_on_boundary(status))
    {
        // Changing direction on a boundary may reverse whether the track
        // will cross the surface; update stored status to match.
        Real3 const& norm = this->normal();

        // Evaluate whether the direction dotted with the surface normal
        // changes (i.e. heading from inside to outside or vice versa).
        auto new_dot = dot_product(norm, newdir);
        if (CELER_UNLIKELY(new_dot == 0))
        {
            CELER_LOG_LOCAL(error)
                << "track direction cannot change to " << newdir
                << " which is perpendicular to the current surface normal";
            this->geo_status(GeoStatus::error);
            return;
        }
        else if ((new_dot > 0) != (dot_product(norm, this->dir()) > 0))
        {
            // The boundary crossing direction has changed! Reverse our
            // plans to change the logical state and move to a new volume.
            this->geo_status(flip_boundary(this->geo_status()));
        }
    }

    dir_ = newdir;
    g4dir_ = to_g4vector(newdir);
    next_step_ = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Get the navigation state.
 */
G4NavigationHistory const* GeantGeoTrackView::nav_history() const
{
    auto* touch = touch_handle_();
    CELER_ASSERT(touch);
    return touch->GetHistory();
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether a next step has been calculated.
 */
CELER_FORCEINLINE bool GeantGeoTrackView::has_next_step() const
{
    return next_step_ != 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track direction is exiting the current volume.
 *
 * If the normal direction is \em perpendicular to the direction of travel,
 * our status should probably be marked as an error (and bump away?)
 */
bool GeantGeoTrackView::is_dir_exiting() const
{
    CELER_EXPECT(this->is_on_boundary());
    return dot_product(normal_, dir_) > 0;
}

//---------------------------------------------------------------------------//
//! Set the geometry tracking status
void GeantGeoTrackView::geo_status(GeoStatus gs)
{
    state_.status[tid_] = gs;
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
