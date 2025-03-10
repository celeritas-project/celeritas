//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CheckedGeoTrackView.t.hh
//! \brief Template definitions for CheckedGeoTrackView.
//---------------------------------------------------------------------------//
#pragma once

#include "CheckedGeoTrackView.hh"

#include <iomanip>

#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the state.
 */
template<class GTV>
CheckedGeoTrackView<GTV>&
CheckedGeoTrackView<GTV>::operator=(GeoTrackInitializer const& init)
{
    GTV::operator=(init);
    CELER_ENSURE(!this->is_outside());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the state from a parent state and new direction.
 */
template<class GTV>
CheckedGeoTrackView<GTV>&
CheckedGeoTrackView<GTV>::operator=(DetailedInitializer const& init)
{
    GTV::operator=(init);
    CELER_ENSURE(!this->is_outside());
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance.
 */
template<class GTV>
real_type CheckedGeoTrackView<GTV>::find_safety()
{
    ++num_safety_;
    return GTV::find_safety();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance up to a given length.
 */
template<class GTV>
real_type CheckedGeoTrackView<GTV>::find_safety(real_type max_safety)
{
    ++num_safety_;
    real_type result = GTV::find_safety(max_safety);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set the direction.
 */
template<class GTV>
void CheckedGeoTrackView<GTV>::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(!this->is_outside());
    GTV::set_dir(newdir);
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next boundary.
 */
template<class GTV>
Propagation CheckedGeoTrackView<GTV>::find_next_step()
{
    CELER_EXPECT(!this->is_outside());
    ++num_intersect_;
    return GTV::find_next_step();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next boundary.
 */
template<class GTV>
Propagation CheckedGeoTrackView<GTV>::find_next_step(real_type distance)
{
    CELER_EXPECT(distance > 0);
    CELER_EXPECT(!this->is_outside());
    ++num_intersect_;
    auto result = GTV::find_next_step(distance);
    if (result.boundary && result.distance > this->safety_tol()
        && !this->is_on_boundary())
    {
        real_type safety = GTV::find_safety(distance);
        CELER_VALIDATE(safety <= result.distance,
                       << std::setprecision(16) << "safety " << safety
                       << " exceeds actual distance " << result.distance
                       << " to boundary at " << this->pos() << " in "
                       << this->volume_id().get());
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume along the current direction.
 */
template<class GTV>
void CheckedGeoTrackView<GTV>::move_internal(real_type step)
{
    CELER_EXPECT(!this->is_outside());
    GTV::move_internal(step);
    CELER_VALIDATE(!this->is_on_boundary() && !this->is_outside()
                       && GTV::find_safety() > 0,
                   << std::setprecision(16)
                   << "zero safety distance after moving " << step << " to "
                   << this->pos());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume to a nearby position.
 */
template<class GTV>
void CheckedGeoTrackView<GTV>::move_internal(Real3 const& pos)
{
    CELER_EXPECT(!this->is_outside());
    real_type orig_safety = (this->is_on_boundary() ? 0 : GTV::find_safety());
    auto orig_pos = this->pos();
    GTV::move_internal(pos);
    CELER_ASSERT(!this->is_on_boundary());
    if (!checked_internal_ && orig_safety > this->safety_tol())
    {
        VolumeId expected = this->volume_id();
        Initializer_t here{this->pos(), this->dir()};
        *this = here;
        CELER_VALIDATE(!this->is_outside(),
                       << std::setprecision(16)
                       << "internal move ends up 'outside' at " << this->pos());
        CELER_VALIDATE(this->volume_id() == expected,
                       << std::setprecision(16)
                       << "volume ID changed during internal move from"
                       << repr(orig_pos) << " to " << repr(this->pos())
                       << ": was " << expected.get() << ", now "
                       << this->volume_id().get());
        checked_internal_ = true;
    }
    if (orig_safety == 0 && !this->is_on_boundary())
    {
        real_type new_safety = GTV::find_safety();
        if (!(new_safety > 0))
        {
            CELER_LOG_LOCAL(warning)
                << "Moved internally from boundary but safety didn't "
                   "increase: volume "
                << this->volume_id().get() << " from " << orig_pos << " to "
                << this->pos() << " (distance: " << distance(orig_pos, pos)
                << ")";
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
template<class GTV>
void CheckedGeoTrackView<GTV>::move_to_boundary()
{
    // Move to boundary
    GTV::move_to_boundary();
    checked_internal_ = false;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
template<class GTV>
void CheckedGeoTrackView<GTV>::cross_boundary()
{
    // Cross boundary
    VolumeId prev = this->volume_id();
    GTV::cross_boundary();
    if (!this->is_outside() && prev == this->volume_id())
    {
        CELER_LOG_LOCAL(warning)
            << "Volume did not change from " << prev.get()
            << " when crossing boundary at " << this->pos();
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
