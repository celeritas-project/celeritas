//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/CheckedGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/Logger.hh"
#include "orange/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Check validity of safety and volume crossings while navigating on CPU.
 *
 * Also count the number of calls to "find distance" and "find safety".
 */
template<class GTV>
class CheckedGeoTrackView : public GTV
{
  public:
    //!@{
    //! \name Type aliases
    using GeoTrackViewT = GTV;
    using Initializer_t = typename GTV::Initializer_t;
    using DetailedInitializer = typename GTV::DetailedInitializer;
    //!@}

  public:
    //! Forward construction arguments to the original track view
    template<class... Args>
    CheckedGeoTrackView(Args&&... args) : GTV(std::forward<Args>(args)...)
    {
    }

    // Initialize the state
    inline CELER_FUNCTION CheckedGeoTrackView&
    operator=(GeoTrackInitializer const& init)
    {
        GTV::operator=(init);
        CELER_ENSURE(!this->is_outside());
        return *this;
    }
    // Initialize the state from a parent state and new direction
    inline CELER_FUNCTION CheckedGeoTrackView&
    operator=(DetailedInitializer const& init)
    {
        GTV::operator=(init);
        CELER_ENSURE(!this->is_outside());
        return *this;
    }

    // Check volume consistency this far from the boundary
    static constexpr real_type safety_tol() { return 1e-4; }

    // Calculate or return the safety up to an infinite distance
    inline real_type find_safety();

    // Calculate or return the safety up to the given distance
    inline real_type find_safety(real_type max_safety);

    // Change the direction
    inline void set_dir(Real3 const&);

    // Find the distance to the next boundary
    inline Propagation find_next_step();

    // Find the distance to the next boundary
    inline Propagation find_next_step(real_type max_distance);

    // Move a linear step fraction
    inline void move_internal(real_type);

    // Move within the safety distance to a specific point
    inline void move_internal(Real3 const& pos);

    // Move to the boundary in preparation for crossing it
    inline void move_to_boundary();

    // Cross from one side of the current surface to the other
    inline void cross_boundary();

    //! Number of calls fo find_next_step
    size_type intersect_count() const { return num_intersect_; }
    //! Number of calls fo find_safety
    size_type safety_count() const { return num_safety_; }
    //! Reset the stepscounter
    void reset_count() { num_intersect_ = num_safety_ = 0; }

  private:
    bool checked_internal_{false};
    size_type num_intersect_{0};
    size_type num_safety_{0};
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class GTV>
CheckedGeoTrackView(GTV&&) -> CheckedGeoTrackView<GTV>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
    if (result.boundary && !this->is_on_boundary())
    {
        real_type safety = GTV::find_safety(distance);
        CELER_VALIDATE(safety <= result.distance,
                       << "safety " << safety << " exceeds actual distance "
                       << result.distance << " to boundary at " << this->pos()
                       << " in " << this->volume_id().get());
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
    GTV::move_internal(pos);
    CELER_ASSERT(!this->is_on_boundary());
    if (!checked_internal_ && orig_safety > this->safety_tol())
    {
        VolumeId expected = this->volume_id();
        Initializer_t here{this->pos(), this->dir()};
        *this = here;
        CELER_VALIDATE(!this->is_outside(),
                       << "internal move ends up 'outside' at " << this->pos());
        CELER_VALIDATE(this->volume_id() == expected,
                       << "volume ID changed during internal move at "
                       << this->pos() << ": was " << expected.get() << ", now "
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
                << this->volume_id().get() << " at " << this->pos();
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
