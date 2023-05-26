//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/SafetyCacheTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "GeoTrackView.hh"
#include "detail/SafetyCacheTrackViewBase.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
//! \cond
// Local convenience declaration
#define CELER_FIF CELER_FORCEINLINE_FUNCTION
//! \endcond

//---------------------------------------------------------------------------//
/*!
 * Wrap geometry functions that use the safety distance.
 *
 * This class stores a persistent safety sphere (origin plus radius) that is
 * updated as needed by the geometry. The view does *not* support direct
 * movement along a straight line but supports moving to arbitrary distances
 * within the safety sphere.
 *
 * Internally the radius has a negative value if safety calculations are not
 * needed for the current track, zero if the safety is on the boundary or has
 * not been updated, and positive for the typical case where the track is not
 * on a boundary.
 *
 * There are two specializations for this class:
 * - mutable (or owning) track view can update the safety and geometry
 * - const reference track view is used for *only* checking the safety
 */
template<class GTV>
class SafetyCacheTrackView : public detail::SafetyCacheTrackViewBase
{
    using Base = detail::SafetyCacheTrackViewBase;

  public:
    //!@{
    //! \name Type aliases
    using StateRef = NativeRef<SafetyCacheStateData>;
    using GeoTrackViewT = GTV;
    using Initializer_t = SafetyCacheInitializer;
    using DetailedInitializer = Base::DetailedInitializer;
    //!@}

  public:
    //! Construct with state data and geometry
    CELER_FUNCTION SafetyCacheTrackView(GeoTrackViewT geo,
                                        StateRef const& state,
                                        TrackSlotId tid)
        : Base{state, tid}, geo_{geo}
    {
    }

    //! Initialize with a flag for safety calculation
    CELER_FUNCTION SafetyCacheTrackView& operator=(Initializer_t const& init)
    {
        Base::operator=(init);
        return *this;
    }

    //! Initialize from another safety cache
    CELER_FUNCTION SafetyCacheTrackView&
    operator=(DetailedInitializer const& init)
    {
        Base::operator=(init);
        return *this;
    }

    //! Whether the safety is being calculated for this track type
    using Base::use_safety;

    //! Whether the given point is inside the safety sphere
    using Base::is_inside;

    //! Calculate the cached safety distance at the current position
    CELER_FIF real_type safety() const
    {
        return Base::calc_safety(geo_.pos());
    }

    // Calculate or return the safety up to the given distance
    inline CELER_FUNCTION real_type find_safety(real_type max_safety);

    // Find the distance to the next boundary
    inline CELER_FUNCTION Propagation find_next_step(real_type max_distance);

    // Move within the safety distance to a specific point
    inline CELER_FUNCTION void move_internal(Real3 const& pos);

    // Move to the boundary in preparation for crossing it
    inline CELER_FUNCTION void move_to_boundary();

    //!@{
    //! Forward state from underlying GeoTrackView
    CELER_FIF Real3 const& pos() const { return geo_.pos(); }
    CELER_FIF Real3 const& dir() const { return geo_.dir(); }
    CELER_FIF bool is_outside() const { return geo_.is_outside(); }
    CELER_FIF bool is_on_boundary() const { return geo_.is_on_boundary(); }
    CELER_FIF void set_dir(Real3 const& d) { return geo_.set_dir(d); }
    //!@}

  private:
    GeoTrackViewT geo_;
};

//---------------------------------------------------------------------------//
//! Read-only specialization for accessing safety
template<class GTV>
class SafetyCacheTrackView<GTV const&> : public detail::SafetyCacheTrackViewBase
{
    using Base = detail::SafetyCacheTrackViewBase;

  public:
    //!@{
    //! \name Type aliases
    using StateRef = NativeRef<SafetyCacheStateData>;
    using GeoTrackViewT = GTV const&;
    using Initializer_t = SafetyCacheInitializer;
    using DetailedInitializer = Base::DetailedInitializer;
    //!@}

  public:
    //! Construct with state data and geometry
    CELER_FUNCTION SafetyCacheTrackView(GeoTrackViewT geo,
                                        StateRef const& state,
                                        TrackSlotId tid)
        : Base{state, tid}, geo_{geo}
    {
    }

    //! Initialize with a flag for safety calculation
    CELER_FUNCTION SafetyCacheTrackView& operator=(Initializer_t const& init)
    {
        Base::operator=(init);
        return *this;
    }

    //! Initialize from another safety cache
    CELER_FUNCTION SafetyCacheTrackView&
    operator=(DetailedInitializer const& init)
    {
        Base::operator=(init);
        return *this;
    }

    //! Whether the safety is being calculated for this track type
    using Base::use_safety;

    //! Whether the given point is inside the safety sphere
    using Base::is_inside;

    //! Calculate the cached safety distance at the current position
    CELER_FIF real_type safety() const
    {
        return Base::calc_safety(geo_.pos());
    }

    //!@{
    //! Forward state from underlying GeoTrackView
    CELER_FIF Real3 const& pos() const { return geo_.pos(); }
    CELER_FIF Real3 const& dir() const { return geo_.dir(); }
    CELER_FIF bool is_outside() const { return geo_.is_outside(); }
    CELER_FIF bool is_on_boundary() const { return geo_.is_on_boundary(); }
    //!@}

  private:
    GeoTrackViewT geo_;
};

#undef CELER_FIF

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class GTV>
CELER_FUNCTION SafetyCacheTrackView(GTV&&,
                                    NativeRef<SafetyCacheStateData> const&,
                                    TrackSlotId tid)
    ->SafetyCacheTrackView<GTV>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Update the safety from the geometry if current is less than this value.
 */
template<class GTV>
inline CELER_FUNCTION real_type
SafetyCacheTrackView<GTV>::find_safety(real_type max_safety)
{
    CELER_EXPECT(max_safety > 0);
    CELER_EXPECT(this->use_safety());

    real_type result = this->calc_safety(geo_.pos());
    if (result >= max_safety)
    {
        return result;
    }

    // Calculate new safety
    result = geo_.find_safety(max_safety);
    CELER_EXPECT(result >= 0);

    // Save it and the position
    this->reset(geo_.pos(), result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Update the safety from the geometry if current is less than this value.
 */
template<class GTV>
inline CELER_FUNCTION Propagation
SafetyCacheTrackView<GTV>::find_next_step(real_type distance)
{
    CELER_EXPECT(distance > 0);
    CELER_EXPECT(this->use_safety());
    if (distance < this->safety())
    {
        return Propagation::from_miss(distance);
    }
    return geo_.find_next_step(distance);
}

//---------------------------------------------------------------------------//
/*!
 * Move within the volume to a nearby position.
 *
 * The other position *must* be within the safety distance.
 */
template<class GTV>
inline CELER_FUNCTION void
SafetyCacheTrackView<GTV>::move_internal(Real3 const& pos)
{
    CELER_EXPECT(this->use_safety());
    CELER_EXPECT(this->is_inside(pos));

    geo_.move_internal(pos);
}

//---------------------------------------------------------------------------//
/*!
 * Move to the next boundary.
 *
 * The other position *must* be within the safety distance.
 */
template<class GTV>
inline CELER_FUNCTION void SafetyCacheTrackView<GTV>::move_to_boundary()
{
    CELER_EXPECT(this->use_safety());

    // Move to boundary
    geo_.move_to_boundary();
    // Reset the safety to zero
    this->reset();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
