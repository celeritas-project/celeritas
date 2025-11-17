//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CheckedGeoTrackView.cc
//---------------------------------------------------------------------------//
#include "CheckedGeoTrackView.hh"

#include <iomanip>
#include <optional>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/math/ArrayUtils.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Initialize the state.
 */
CheckedGeoTrackView&
CheckedGeoTrackView::operator=(GeoTrackInitializer const& init)
{
    CELER_VALIDATE(is_soft_unit_vector(init.dir),
                   << "cannot initialize with a non-unit direction "
                   << repr(init.dir));

    *t_ = init;
    CELER_VALIDATE(!t_->failed(),
                   << "failed to initialize at " << repr(init.pos) << " along "
                   << repr(init.dir));
    CELER_VALIDATE(!t_->is_outside(),
                   << "initialized outside at " << repr(init.pos) << " along "
                   << repr(init.dir));
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance.
 */
real_type CheckedGeoTrackView::find_safety()
{
    ++num_safety_;
    return t_->find_safety();
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the safety distance up to a given length.
 */
real_type CheckedGeoTrackView::find_safety(real_type max_safety)
{
    ++num_safety_;
    real_type result = t_->find_safety(max_safety);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Set the direction.
 */
void CheckedGeoTrackView::set_dir(Real3 const& newdir)
{
    CELER_EXPECT(!t_->is_outside());
    CELER_VALIDATE(is_soft_unit_vector(newdir),
                   << "cannot change to a non-unit direction " << repr(newdir));
    t_->set_dir(newdir);
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next boundary.
 */
Propagation CheckedGeoTrackView::find_next_step()
{
    CELER_VALIDATE(!t_->is_outside(), << "cannot find next step from outside");
    ++num_intersect_;
    return t_->find_next_step();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next boundary.
 */
Propagation CheckedGeoTrackView::find_next_step(real_type distance)
{
    CELER_VALIDATE(distance > 0, << "invalid step maximum " << repr(distance));
    CELER_VALIDATE(!t_->is_outside(), << "cannot find next step from outside");
    ++num_intersect_;
    auto result = t_->find_next_step(distance);
    if (result.boundary && result.distance > this->safety_tol()
        && !t_->is_on_boundary())
    {
        real_type safety = t_->find_safety(distance);
        CELER_VALIDATE(safety <= result.distance,
                       << std::setprecision(16) << "safety " << safety
                       << " exceeds actual distance " << result.distance
                       << " to boundary at " << t_->pos() << " in "
                       << t_->impl_volume_id().get());
    }
    CELER_VALIDATE(result.distance <= distance,
                   << "return distance " << result.distance
                   << " exceeds maximum search value " << distance);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume along the current direction.
 */
void CheckedGeoTrackView::move_internal(real_type step)
{
    CELER_EXPECT(!t_->is_outside());
    t_->move_internal(step);
    CELER_VALIDATE(
        !t_->is_on_boundary() && !t_->is_outside() && t_->find_safety() >= 0,
        << std::setprecision(16) << "zero safety distance after moving "
        << step << " to " << t_->pos());
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume to a nearby position.
 */
void CheckedGeoTrackView::move_internal(Real3 const& pos)
{
    CELER_EXPECT(!t_->is_outside());
    real_type orig_safety = (t_->is_on_boundary() ? 0 : t_->find_safety());
    auto orig_pos = t_->pos();
    t_->move_internal(pos);
    CELER_ASSERT(!t_->is_on_boundary());
    if (!checked_internal_ && orig_safety > this->safety_tol())
    {
        ImplVolumeId expected = t_->impl_volume_id();
        Initializer_t here{t_->pos(), t_->dir()};
        *t_ = here;
        CELER_VALIDATE(!t_->is_outside(),
                       << std::setprecision(16)
                       << "internal move ends up 'outside' at " << t_->pos());
        CELER_VALIDATE(t_->impl_volume_id() == expected,
                       << std::setprecision(16)
                       << "volume ID changed during internal move from"
                       << repr(orig_pos) << " to " << repr(t_->pos())
                       << ": was " << expected.get() << ", now "
                       << t_->impl_volume_id().get());
        checked_internal_ = true;
    }
    if (orig_safety == 0 && !t_->is_on_boundary())
    {
        real_type new_safety = t_->find_safety();
        if (!(new_safety > 0))
        {
            CELER_LOG_LOCAL(warning)
                << "Moved internally from boundary but safety didn't "
                   "increase: volume "
                << t_->impl_volume_id().get() << " from " << repr(orig_pos)
                << " to " << repr(t_->pos())
                << " (distance: " << repr(distance(orig_pos, pos)) << ")";
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
void CheckedGeoTrackView::move_to_boundary()
{
    // Move to boundary
    t_->move_to_boundary();
    checked_internal_ = false;
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 */
void CheckedGeoTrackView::cross_boundary()
{
    CELER_VALIDATE(t_->is_on_boundary(),
                   << "attempted to cross boundary without being on boundary");

    // Capture pre-crossing normal if checking is enabled
    std::optional<Real3> pre_crossing_normal;
    if (check_normal_ && !t_->is_outside())
    {
        pre_crossing_normal = t_->normal();
    }

    // Cross boundary
    t_->cross_boundary();
    CELER_VALIDATE(t_->is_on_boundary(),
                   << std::setprecision(16)
                   << "internal move ends up 'outside' at " << t_->pos());

    // Verify post-crossing normal if checking is enabled
    if (pre_crossing_normal && !t_->is_outside())
    {
        auto post_norm = t_->normal();
        CELER_VALIDATE(
            soft_equal(std::fabs(dot_product(*pre_crossing_normal, post_norm)),
                       real_type{1}),
            << std::setprecision(16)
            << "Normal is not consistent at boundary: pre-crossing "
            << *pre_crossing_normal << ", post-crossing " << post_norm);

        // Check for tangent crossing warning
        if (soft_zero(dot_product(t_->dir(), post_norm)))
        {
            CELER_LOG(warning)
                << "Crossed into " << t_->impl_volume_id().get()
                << " at a tangent; traveling along " << repr(t_->dir())
                << ", normal is " << repr(post_norm);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
