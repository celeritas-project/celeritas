//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VgBasicTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#if CELERITAS_VECGEOM_VERSION < 0x020100
#    error "This file requires VecGeom 2.1+"
#endif

#include <VecGeom/navigation/NavView.h>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArraySoftUnit.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/Types.hh"

#include "VecgeomData.hh"
#include "VecgeomTypes.hh"

#if !CELER_DEVICE_COMPILE
#    include "corecel/io/Logger.hh"
#    include "corecel/io/Repr.hh"
#    include "geocel/detail/LengthUnits.hh"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap basic navigation functionality from VecGeom.
 *
 * For a description of ordering requirements, see:
 * \sa OrangeTrackView
 *
 * \code
    VgBasicTrackView geom(vg_params_ref, vg_state_ref, trackslot_id);
   \endcode
 *
 * The "next distance" is cached as part of `find_next_step`, but it is only
 * used when the immediate next call is `move_to_boundary`.
 */
class VgBasicTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = GeoTrackInitializer;
    using ParamsRef = NativeCRef<VecgeomParamsData>;
    using StateRef = NativeRef<VecgeomStateData>;
    using NavView = vecgeom::NavView;
    using OpaquePath = NavView::OpaquePath;
    using real_type = vg_real_type;
    using ImplVolInstanceId = VgVolumeInstanceId;
    //!@}

  public:
    // Construct from persistent and state data
    inline CELER_FUNCTION VgBasicTrackView(
        ParamsRef const& data, StateRef const& stateview, TrackSlotId tid);

    // Initialize the state
    inline CELER_FUNCTION VgBasicTrackView& operator=(
        Initializer_t const& init);

    //// ACCESSORS ////

    //!@{
    //! State accessors
    CELER_FORCEINLINE_FUNCTION Real3 const& pos() const { return pos_; }
    CELER_FORCEINLINE_FUNCTION Real3 const& dir() const { return dir_; }
    //!@}

    // Get the canonical volume ID in the current impl volume
    inline CELER_FUNCTION VolumeId volume_id() const;
    // Get the ID of the current volume instance
    inline CELER_FUNCTION VolumeInstanceId volume_instance_id() const;
    // Get the depth in the geometry hierarchy
    inline CELER_FUNCTION VolumeLevelId volume_level() const;
    // Get the volume instance ID for all levels
    inline CELER_FUNCTION void volume_instance_id(
        Span<VolumeInstanceId> levels) const;
    // Visit every volume instance in the track's path, including world
    template<class F>
    inline CELER_FUNCTION void foreach_volume_path(F&& visit) const;

    // Get the current volume's ID
    inline CELER_FUNCTION ImplVolumeId impl_volume_id() const;

    // Get the opaque nav path (index or tuple)
    inline CELER_FUNCTION OpaquePath opaque_path() const;

    // Whether the track is outside the valid geometry region
    inline CELER_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    inline CELER_FUNCTION bool is_on_boundary() const;
    //! Whether the last operation resulted in an error
    CELER_FORCEINLINE_FUNCTION bool failed() const { return failed_; }
    // Get the normal vector of the current surface
    inline CELER_FUNCTION Real3 normal() const;

    //// OPERATIONS ////

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Find the safety at the current position (infinite max)
    inline CELER_FUNCTION real_type find_safety();

    // Find the safety at the current position up to a maximum step distance
    inline CELER_FUNCTION real_type find_safety(real_type max_step);

    // Move to the boundary in preparation for crossing it
    inline CELER_FUNCTION void move_to_boundary();

    // Move within the volume
    inline CELER_FUNCTION void move_internal(real_type step);

    // Move within the volume to a specific point
    inline CELER_FUNCTION void move_internal(Real3 const& pos);

    // Cross from one side of the current surface to the other
    inline CELER_FUNCTION void cross_boundary();

    // Change direction
    inline CELER_FUNCTION void set_dir(Real3 const& newdir);

  private:
    //// TYPES ////

    using VgLogVol = VgLogicalVolume<MemSpace::native>;
    using VgPlacedVol = VgPlacedVolume<MemSpace::native>;

    //// DATA ////

    //! Shared/persistent geometry data
    ParamsRef const& params_;
    StateRef const& state_;
    TrackSlotId tid_;

    //!@{
    //! Referenced thread-local data
    VgNavState& vgstate_;
    VgNavState& vgnext_;
    Real3& pos_;
    Real3& dir_;

    //!@}

    // Temporary data
    bool failed_{false};

    //// HELPER CLASSES ////

    struct LocalNavData;
    class LocalNav;

    //// HELPER FUNCTIONS ////

    // Whether the next distance-to-boundary is to a surface
    inline CELER_FUNCTION bool is_next_boundary() const;

    // Get a reference to the current volume instance
    inline CELER_FUNCTION VgPlacedVol const& physical_volume() const;

    // Get a reference to the current volume
    inline CELER_FUNCTION VgLogVol const& logical_volume() const;

    // Create a temporary navigator
    inline CELER_FUNCTION LocalNav make_nav() const;
};

//---------------------------------------------------------------------------//
// LOCAL NAV
// This will be simplified when the upstream NavView supports true Span.
//---------------------------------------------------------------------------//

struct VgBasicTrackView::LocalNavData
{
    VgReal3 temp_pos;
    VgReal3 temp_dir;
};

class VgBasicTrackView::LocalNav : public LocalNavData, public vecgeom::NavView
{
  public:
    explicit LocalNav(VgBasicTrackView& vtv)
        : LocalNavData{to_vgvector(vtv.pos_), to_vgvector(vtv.dir_)}
        , NavView{vtv.vgstate_, vtv.vgnext_, this->temp_pos, this->temp_dir}
    {
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from persistent and state data.
 */
CELER_FUNCTION VgBasicTrackView::VgBasicTrackView(
    ParamsRef const& params, StateRef const& states, TrackSlotId tid)
    : params_(params)
    , state_(states)
    , tid_(tid)
    , vgstate_{states.state[tid]}
    , vgnext_{states.next_state[tid]}
    , pos_(states.pos[tid])
    , dir_(states.dir[tid])
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 *
 * If a valid parent ID is provided, the state is constructed from a direction
 * and a copy of the parent state.  This is a faster method of creating
 * secondaries from a parent that has just been absorbed, or when filling in an
 * empty track from a parent that is still alive.
 *
 * Otherwise, the state is initialized from a starting location and direction,
 * which is expensive.
 */
CELER_FUNCTION VgBasicTrackView& VgBasicTrackView::operator=(
    Initializer_t const& init)
{
    CELER_EXPECT(is_soft_unit_vector(init.dir));
    failed_ = false;

    // Copy dir/pos: make_nav currently does array -> vgreal3 conversion
    dir_ = init.dir;
    pos_ = init.pos;
    auto nav = this->make_nav();
    if (init.parent)
    {
        VgBasicTrackView other(params_, state_, init.parent);
        nav.Initialize(
            vecgeom::FullPath{other.opaque_path(), other.is_on_boundary()},
            nav.temp_pos,
            nav.temp_dir);
    }
    else
    {
        nav.Initialize(vecgeom::UnknownPath{}, nav.temp_pos, nav.temp_dir);
    }

    if (CELER_UNLIKELY(nav.IsOutside()))
    {
#if !CELER_DEVICE_COMPILE
        auto msg = CELER_LOG_LOCAL(error);
        msg << "Failed to initialize geometry state at " << repr(pos_) << ' '
            << lengthunits::native_label;
#endif
        failed_ = true;
    }

    CELER_ENSURE(this->dir() == to_array(nav.GetDirection()));
    CELER_ENSURE(this->pos() == to_array(nav.GetPosition()));
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FORCEINLINE_FUNCTION ImplVolumeId VgBasicTrackView::impl_volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return id_cast<ImplVolumeId>(this->logical_volume().id());
}

//---------------------------------------------------------------------------//
/*!
 * After 'find_next_step', the next straight-line surface.
 */
CELER_FORCEINLINE_FUNCTION auto VgBasicTrackView::opaque_path() const
    -> OpaquePath
{
    return this->make_nav().GetOpaquePath();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool VgBasicTrackView::is_outside() const
{
    return vgstate_.IsOutside();
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is on the boundary of a volume.
 */
CELER_FUNCTION bool VgBasicTrackView::is_on_boundary() const
{
    return vgstate_.IsOnBoundary();
}

//---------------------------------------------------------------------------//
/*!
 * Get the surface normal of the boundary the track is currently on.
 */
CELER_FUNCTION Real3 VgBasicTrackView::normal() const
{
    // FIXME: temporarily return a bogus but valid surface normal
    return this->dir();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation VgBasicTrackView::find_next_step(real_type max_step)
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(max_step > 0);

    using Kind = vecgeom::NavFindResultKind;

    auto found = this->make_nav().FindNextBoundary(max_step);

    Propagation result;
    if (found.GetKind() == Kind::hit)
    {
        result.distance = found.GetDistance();
        result.boundary = true;
        CELER_ASSERT(found.GetDistance() < max_step);
    }
    else if (found.GetKind() == Kind::miss)
    {
        result.distance = max_step;
        result.boundary = false;
    }
    else if (found.GetKind() == Kind::reentrant)
    {
        result.distance = 0;
        result.boundary = true;
    }
    else if (CELER_UNLIKELY(found.GetKind() == Kind::error))
    {
#if !CELER_DEVICE_COMPILE
        auto msg = CELER_LOG_LOCAL(debug);
        msg << "Failed to find next step at " << repr(pos_) << ' '
            << lengthunits::native_label << " along " << repr(dir_);
#endif
        failed_ = true;
        result.distance = 0;
        result.boundary = true;
    }
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }

    CELER_ENSURE(result.distance >= 0 && result.distance <= max_step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety at the current position.
 */
CELER_FUNCTION real_type VgBasicTrackView::find_safety()
{
    return this->find_safety(vecgeom::kInfLength);
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety at the current position up to a maximum distance.
 *
 * The safety within a step is only needed up to the end of the physics step
 * length.
 */
CELER_FUNCTION real_type VgBasicTrackView::find_safety(real_type max_radius)
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(!this->is_on_boundary());
    CELER_EXPECT(max_radius > 0);

    auto safety = this->make_nav().FindSafety(max_radius);
    CELER_ENSURE(safety >= 0 && safety <= max_radius);
    return safety;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary but don't cross yet.
 *
 * Note that, unlike the regular move_internal, we allow zero distance in this
 * case since it results in a change in state (though that \em may allow a
 * cycle).
 */
CELER_FUNCTION void VgBasicTrackView::move_to_boundary(real_type dist)
{
    CELER_EXPECT(dist >= 0);
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(this->is_next_boundary());

    auto nav = this->make_nav();
    nav.MoveToBoundary(dist);
    pos_ = to_array(nav.GetPosition());
    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary.
 */
CELER_FUNCTION void VgBasicTrackView::cross_boundary()
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(this->is_on_boundary());
    CELER_EXPECT(this->is_next_boundary());

    this->make_nav().CrossBoundary();
    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume.
 *
 * The straight-line distance *must* be less than the distance to the
 * boundary.
 */
CELER_FUNCTION void VgBasicTrackView::move_internal(real_type dist)
{
    CELER_EXPECT(dist > 0);
    CELER_EXPECT(!this->is_next_boundary());

    auto nav = this->make_nav();
    nav.MoveInternal(dist);
    pos_ = to_array(nav.GetPosition());

    CELER_ENSURE(!this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume to a nearby point.
 *
 * \warning It's up to the caller to make sure that the position is
 * "nearby" and within the same volume.
 */
CELER_FUNCTION void VgBasicTrackView::move_internal(Real3 const& pos)
{
    pos_ = pos;
    auto nav = this->make_nav();
    nav.MoveToBoundary(next_step_);
    pos_ = to_array(nav.GetPosition());
    CELER_ENSURE(!this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Change the track's direction.
 *
 * This happens after a scattering event or movement inside a magnetic field.
 * It resets the calculated distance-to-boundary.
 */
CELER_FUNCTION void VgBasicTrackView::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(is_soft_unit_vector(newdir));
    auto nav = this->make_nav();
    nav.ChangeDirection(to_vgvector(newdir));
    dir_ = to_array(nav.GetDirection());
}

//---------------------------------------------------------------------------//
// PRIVATE MEMBER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Whether the calculated next step will take track to next boundary.
 */
CELER_FUNCTION bool VgBasicTrackView::is_next_boundary() const
{
    return vgnext_.IsOnBoundary();
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume.
 */
CELER_FUNCTION auto VgBasicTrackView::physical_volume() const
    -> VgPlacedVol const&
{
    VgPlacedVol const* physvol_ptr = vgstate_.Top();
    CELER_ENSURE(physvol_ptr);
    return *physvol_ptr;
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the current volume, or to world volume if outside.
 */
CELER_FUNCTION auto VgBasicTrackView::logical_volume() const -> VgLogVol const&
{
    return *this->physical_volume().GetLogicalVolume();
}

// Create a temporary navigator
CELER_FUNCTION auto VgBasicTrackView::make_nav() const -> LocalNav
{
    return LocalNav{const_cast<VgBasicTrackView&>(*this)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
