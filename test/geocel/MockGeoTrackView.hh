//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
 * doesn't have full functionality. Consider using ORANGE geometery for more
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
    CELER_FORCEINLINE_FUNCTION VolumeId volume_id() const;

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
    GeoTrackInitializer state_;
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
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get the volume ID in the current cell.
 *
 * XXX this doesn't correctly handle the logical change in state for boundary
 * crossing.
 */
CELER_FUNCTION VolumeId MockGeoTrackView::volume_id() const
{
    CELER_EXPECT(!this->is_outside());
    return VolumeId{static_cast<size_type>(std::floor(this->z()))};
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is outside the valid geometry region.
 */
CELER_FUNCTION bool MockGeoTrackView::is_outside() const
{
    return this->z() <= 0;
}

//---------------------------------------------------------------------------//
/*!
 * Whether the track is on the boundary of a volume.
 */
CELER_FUNCTION bool MockGeoTrackView::is_on_boundary() const
{
    return soft_mod(this->z(), real_type{1});
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
    real_type next_surf{100000};
    if (w > 0)
    {
        next_surf = (1 - std::fmod(z, real_type{1})) / w;
    }
    else if (w < 0)
    {
        next_surf = std::fmod(z, real_type{1}) / -w;
    }
    next_step_ = std::fmin(next_surf, max_step);

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
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
