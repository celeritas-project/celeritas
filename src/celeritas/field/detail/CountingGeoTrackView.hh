//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/CountingGeoTrackView.hh
//! \sa test/celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/math/Algorithms.hh"
#include "geocel/Types.hh"

#include "../FieldPropagationCounters.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Count geometry navigation calls made during field propagation.
 *
 * This wraps a geometry track view (by value or reference) and forwards the
 * subset of the geometry interface used by \c FieldPropagator. It is a
 * device-compatible counterpart to the counters in \c CheckedGeoTrackView.
 *
 * An optional "oracle" safety sphere, calculated by the caller before
 * propagating, measures how many boundary searches a safety-based propagator
 * could avoid. A search is counted as skippable if its end point, the current
 * position plus the maximum search distance along the current direction, is
 * inside the sphere. Since the sphere is convex and contains the start point,
 * it then contains the whole search segment.
 */
template<class GTV>
class CountingGeoTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<real_type, 3>;
    //!@}

  public:
    //! Construct with the geometry track view and the counters to increment
    CELER_FUNCTION CountingGeoTrackView(GTV&& geo,
                                        FieldPropagationCounters* counters)
        : geo_{::celeritas::forward<GTV>(geo)}, c_{counters}
    {
        CELER_EXPECT(c_);
    }

    // Set a sphere known to be free of boundaries
    inline CELER_FUNCTION void set_safety_sphere(Real3 const& center,
                                                 real_type radius);

    //!@{
    //! \name Forwarded state accessors
    CELER_FUNCTION decltype(auto) pos() const { return geo_.pos(); }
    CELER_FUNCTION decltype(auto) dir() const { return geo_.dir(); }
    CELER_FUNCTION decltype(auto) volume_id() const
    {
        return geo_.volume_id();
    }
    CELER_FUNCTION bool is_on_boundary() const
    {
        return geo_.is_on_boundary();
    }
    CELER_FUNCTION bool failed() const { return geo_.failed(); }
    //!@}

    //!@{
    //! \name Forwarded state mutators
    CELER_FUNCTION void set_dir(Real3 const& dir) { geo_.set_dir(dir); }
    CELER_FUNCTION void move_internal(Real3 const& pos)
    {
        geo_.move_internal(pos);
    }
    CELER_FUNCTION void move_to_boundary() { geo_.move_to_boundary(); }
    //!@}

    // Find the distance to the next boundary and count the search
    inline CELER_FUNCTION Propagation find_next_step(real_type max_step);

    // Find the safety distance and count the search
    inline CELER_FUNCTION real_type find_safety();

  private:
    GTV geo_;
    FieldPropagationCounters* c_;
    Real3 sphere_center_{0, 0, 0};
    real_type sphere_radius_{0};
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class GTV>
CELER_CTAD_FUNCTION CountingGeoTrackView(GTV&&, FieldPropagationCounters*)
    -> CountingGeoTrackView<GTV>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Set a sphere known to be free of boundaries.
 *
 * A zero radius disables the skippable search counter.
 */
template<class GTV>
CELER_FUNCTION void CountingGeoTrackView<GTV>::set_safety_sphere(
    Real3 const& center, real_type radius)
{
    CELER_EXPECT(radius >= 0);
    sphere_center_ = center;
    sphere_radius_ = radius;
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next boundary and count the search.
 */
template<class GTV>
CELER_FUNCTION Propagation CountingGeoTrackView<GTV>::find_next_step(
    real_type max_step)
{
    ++c_->intersects;
    if (sphere_radius_ > 0)
    {
        // Check whether the end of the search segment is inside the sphere
        Real3 const& pos = geo_.pos();
        Real3 const& dir = geo_.dir();
        real_type dist_sq = 0;
        for (int i = 0; i < 3; ++i)
        {
            dist_sq += ipow<2>(pos[i] + max_step * dir[i] - sphere_center_[i]);
        }
        if (dist_sq <= ipow<2>(sphere_radius_))
        {
            ++c_->skippable_intersects;
        }
    }

    Propagation result = geo_.find_next_step(max_step);
    if (result.boundary)
    {
        ++c_->intersect_hits;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the safety distance and count the search.
 */
template<class GTV>
CELER_FUNCTION real_type CountingGeoTrackView<GTV>::find_safety()
{
    ++c_->safeties;
    return geo_.find_safety();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
