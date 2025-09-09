//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/MockGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/Assert.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/SoftEqual.hh"
#include "geocel/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Mock track view for testing.
 *
 * The mock track view has boundaries at every integer z, with a volume ID
 * that's equal to the floor of the current position. The geometry is outside
 * for z < 0.
 *
 * \note This class is only used by the raytracer test at present, so it
 * doesn't have full functionality. Consider using ORANGE geometry for more
 * complex test cases.
 */
class MockGeoTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = GeoTrackInitializer;
    //!@}

  public:
    //! Default has uninitialized state
    MockGeoTrackView() = default;

    // Initialize the state
    inline CELER_FUNCTION MockGeoTrackView&
    operator=(Initializer_t const& init);

    //!@{
    //! State accessors
    CELER_FORCEINLINE_FUNCTION Real3 const& pos() const { return state_.pos; }
    CELER_FORCEINLINE_FUNCTION Real3 const& dir() const { return state_.dir; }
    //!@}

    // Get the volume ID in the current cell.
    CELER_FORCEINLINE_FUNCTION ImplVolumeId impl_volume_id() const;

    // Whether the track is outside the valid geometry region
    CELER_FORCEINLINE_FUNCTION bool is_outside() const;
    // Whether the track is exactly on a surface
    CELER_FORCEINLINE_FUNCTION bool is_on_boundary() const;

    // Find the distance to the next boundary, up to and including a step
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Move to the boundary in preparation for crossing it
    inline CELER_FUNCTION void move_to_boundary();

    // Move within the volume
    inline CELER_FUNCTION void move_internal(real_type step);

    // Cross from one side of the current surface to the other
    inline CELER_FUNCTION void cross_boundary();

    //! Number of times initialized with a GeoTrackInitializer
    int init_count() const { return init_count_; }

  private:
    // Physical state
    GeoTrackInitializer state_;

    // Logical state
    int volume_id_{-1};
    bool on_boundary_{false};

    real_type next_step_{};
    int init_count_{0};

    // Z value
    CELER_FORCEINLINE_FUNCTION real_type z() const { return state_.pos[2]; }
};

//---------------------------------------------------------------------------//
/*!
 * Construct the state.
 */
CELER_FUNCTION MockGeoTrackView&
MockGeoTrackView::operator=(Initializer_t const& init)
{
    ++init_count_;
    this->state_ = init;
    volume_id_ = static_cast<int>(std::floor(this->z()));
    on_boundary_ = false;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 */
CELER_FUNCTION ImplVolumeId MockGeoTrackView::impl_volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return id_cast<ImplVolumeId>(volume_id_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool MockGeoTrackView::is_outside() const
{
    return volume_id_ < 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is on the boundary of a volume.
 */
CELER_FUNCTION bool MockGeoTrackView::is_on_boundary() const
{
    return on_boundary_;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
CELER_FUNCTION Propagation MockGeoTrackView::find_next_step(real_type max_step)
{
    CELER_EXPECT(!this->is_outside());
    CELER_EXPECT(max_step > 0);

    real_type z = state_.pos[2];
    real_type w = state_.dir[2];

    int next_z_plane = volume_id_ + (w > 0);
    next_step_
        = std::fmin((static_cast<real_type>(next_z_plane) - z) / w, max_step);

    Propagation result;
    result.boundary = next_step_ < max_step;
    result.distance = next_step_;

    CELER_ENSURE(result.distance > 0);
    CELER_ENSURE(result.distance <= max_step);
    CELER_ENSURE(result.boundary || result.distance == max_step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary but don't cross yet.
 */
CELER_FUNCTION void MockGeoTrackView::move_to_boundary()
{
    // Move next step
    this->move_internal(next_step_);
    on_boundary_ = true;
    next_step_ = 0;

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Cross from one side of the current surface to the other.
 *
 * The position *must* be on the boundary following a move-to-boundary.
 */
CELER_FUNCTION void MockGeoTrackView::cross_boundary()
{
    CELER_EXPECT(this->is_on_boundary());

    // Update volume ID
    real_type w = state_.dir[2];
    volume_id_ += (w > 0 ? 1 : -1);

    // Make sure we're exactly on the value
    this->state_.pos[2] = std::round(this->state_.pos[2]);

    CELER_ENSURE(this->is_on_boundary());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the current volume.
 *
 * The straight-line distance *must* be less than the distance to the
 * boundary.
 */
CELER_FUNCTION void MockGeoTrackView::move_internal(real_type dist)
{
    CELER_EXPECT(dist > 0 && dist <= next_step_);

    axpy(dist, state_.dir, &state_.pos);
    next_step_ -= dist;
    on_boundary_ = false;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
