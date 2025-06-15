//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/RectArrayTracker.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/grid/NonuniformGrid.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/OrangeData.hh"

#include "detail/RaggedRightIndexer.hh"
#include "detail/Types.hh"
#include "detail/Utils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Track a particle within an axes-aligned rectilinear grid.
 */
class RectArrayTracker
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsRef = NativeCRef<OrangeParamsData>;
    using Initialization = detail::Initialization;
    using Intersection = detail::Intersection;
    using LocalState = detail::LocalState;
    using Grid = NonuniformGrid<real_type>;
    using VolumeIndexer = HyperslabIndexer<3>;
    using VolumeInverseIndexer = HyperslabInverseIndexer<3>;
    using SurfaceIndexer = detail::RaggedRightIndexer<3>;
    using SurfaceInverseIndexer = detail::RaggedRightInverseIndexer<3>;
    using Coords = Array<size_type, 3>;
    //!@}

  public:
    // Construct with parameters (unit definitions and this one's ID)
    inline CELER_FUNCTION
    RectArrayTracker(ParamsRef const& params, RectArrayId rid);

    //// ACCESSORS ////

    //! Number of local volumes
    CELER_FUNCTION LocalVolumeId::size_type num_volumes() const
    {
        return record_.daughters.size();
    }

    //! Number of local surfaces
    CELER_FUNCTION LocalSurfaceId::size_type num_surfaces() const
    {
        size_type num_surfs = 0;
        for (auto ax : range(Axis::size_))
        {
            num_surfs += record_.dims[to_int(ax)] + 1;
        }
        return num_surfs;
    }

    // DaughterId of universe embedded in a given volume
    inline CELER_FUNCTION DaughterId daughter(LocalVolumeId vol) const;

    ////// OPERATIONS ////

    // Find the local volume from a position
    inline CELER_FUNCTION Initialization
    initialize(LocalState const& state) const;

    // Calculate distance-to-intercept for the next surface
    inline CELER_FUNCTION Intersection intersect(LocalState const& state) const;

    // Calculate distance-to-intercept for the next surface, with max distance
    inline CELER_FUNCTION Intersection intersect(LocalState const& state,
                                                 real_type max_dist) const;

    // Find the local volume given a post-crossing state
    inline CELER_FUNCTION Initialization
    cross_boundary(LocalState const& state) const;

    // Calculate closest distance to a surface in any direction
    inline CELER_FUNCTION real_type safety(Real3 const& pos,
                                           LocalVolumeId vol) const;

    // Calculate the local surface normal
    inline CELER_FUNCTION Real3 normal(Real3 const& pos,
                                       LocalSurfaceId surf) const;

  private:
    //// DATA ////
    ParamsRef const& params_;
    RectArrayRecord const& record_;

    //// METHODS ////

    // Calculate distance-to-intercept for the next surface.
    template<class F>
    inline CELER_FUNCTION Intersection intersect_impl(LocalState const&,
                                                      F) const;

    // Find the index of axis (x/y/z) we are about to cross
    inline CELER_FUNCTION size_type find_surface_axis_idx(LocalSurfaceId s) const;

    // Create Grid object for a given axis
    inline CELER_FUNCTION Grid make_grid(Axis ax) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with reference to persistent parameter data.
 */
CELER_FUNCTION
RectArrayTracker::RectArrayTracker(ParamsRef const& params, RectArrayId rid)
    : params_(params), record_(params.rect_arrays[rid])
{
    CELER_EXPECT(params_);
}

//---------------------------------------------------------------------------//
/*!
 * Find the local volume from a position.
 *
 * To avoid edge cases and inconsistent logical/physical states, it is
 * prohibited to initialize from an arbitrary point directly onto a surface.
 */
CELER_FUNCTION auto RectArrayTracker::initialize(LocalState const& state) const
    -> Initialization
{
    CELER_EXPECT(params_);
    CELER_EXPECT(!state.surface && !state.volume);

    Coords coords;

    for (auto ax : range(Axis::size_))
    {
        auto const& pos = state.pos[to_int(ax)];
        auto grid = this->make_grid(ax);

        if (pos < grid.front() || pos > grid.back())
        {
            // Outside the rect array
            return {};
        }
        else
        {
            size_type index = grid.find(pos);
            if (grid[index] == pos)
            {
                // Initialization exactly on an edge is prohibited
                return {};
            }
            coords[to_int(ax)] = index;
        }
    }

    VolumeIndexer to_index(record_.dims);
    return {LocalVolumeId{to_index(coords)}, {}};
}

//---------------------------------------------------------------------------//
/*!
 * Find the local volume given a post-crossing state.
 */
CELER_FUNCTION auto
RectArrayTracker::cross_boundary(LocalState const& state) const
    -> Initialization
{
    CELER_EXPECT(state.surface && state.volume);

    // Find the coords of the current volume
    VolumeIndexer to_index(record_.dims);
    VolumeInverseIndexer to_coords(record_.dims);
    auto coords = to_coords(state.volume.unchecked_get());
    auto ax_idx = this->find_surface_axis_idx(state.surface.id());

    // Due to surface deduplication, crossing out of rect arrays is not
    // possible
    CELER_ASSERT(
        !(coords[ax_idx] == 0 && state.surface.sense() == Sense::inside)
        && !(coords[ax_idx] == record_.dims[ax_idx] - 1
             && state.surface.sense() == Sense::outside));

    // Value for incrementing the axial coordinate upon crossing
    // NOTE: that surface sense is currently the POST crossing value
    int inc = (state.surface.sense() == Sense::outside) ? 1 : -1;

    coords[ax_idx] += inc;
    return {LocalVolumeId(to_index(coords)), state.surface};
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance-to-intercept for the next surface.
 */
CELER_FUNCTION auto RectArrayTracker::intersect(LocalState const& state) const
    -> Intersection
{
    Intersection result = this->intersect_impl(state, detail::IsFinite{});
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate distance-to-intercept for the next surface, with max distance.
 */
CELER_FUNCTION auto
RectArrayTracker::intersect(LocalState const& state, real_type max_dist) const
    -> Intersection
{
    CELER_EXPECT(max_dist > 0);
    Intersection result
        = this->intersect_impl(state, detail::IsNotFurtherThan{max_dist});
    if (!result)
    {
        result.distance = max_dist;
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate nearest distance to a surface in any direction.
 *
 * On an axis-aligned rectlinear grid the minimum distance to any surface is
 * always always occurs along a line parallel to an axis.
 */
CELER_FUNCTION real_type RectArrayTracker::safety(Real3 const& pos,
                                                  LocalVolumeId vol_id) const
{
    CELER_EXPECT(vol_id && vol_id.get() < this->num_volumes());

    VolumeInverseIndexer to_coords(record_.dims);
    auto coords = to_coords(vol_id.unchecked_get());

    real_type min_dist = numeric_limits<real_type>::infinity();

    for (auto ax : range(Axis::size_))
    {
        auto grid = this->make_grid(ax);
        for (auto i : range(2))
        {
            auto target_coord = coords[to_int(ax)] + i;
            real_type target = grid[target_coord];
            min_dist = min(min_dist, std::fabs(pos[to_int(ax)] - target));
        }
    }

    CELER_ENSURE(min_dist >= 0
                 && min_dist < numeric_limits<real_type>::infinity());
    return min_dist;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the local surface normal.
 */
CELER_FUNCTION auto
RectArrayTracker::normal(Real3 const&, LocalSurfaceId surf) const -> Real3
{
    CELER_EXPECT(surf && surf.get() < this->num_surfaces());
    size_type ax = this->find_surface_axis_idx(surf);

    Real3 normal{0, 0, 0};
    normal[ax] = 1;

    return normal;
}

//---------------------------------------------------------------------------//
/*!
 * DaughterId of universe embedded in a given volume.
 */
CELER_FORCEINLINE_FUNCTION DaughterId
RectArrayTracker::daughter(LocalVolumeId vol) const
{
    CELER_EXPECT(vol && vol.get() < this->num_volumes());
    return record_.daughters[vol];
}

//---------------------------------------------------------------------------//
// PRIVATE INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate distance-to-intercept for the next surface.
 */
template<class F>
CELER_FUNCTION auto
RectArrayTracker::intersect_impl(LocalState const& state, F is_valid) const
    -> Intersection
{
    CELER_EXPECT(state.volume && !state.temp_sense.empty());

    auto coords
        = VolumeInverseIndexer{record_.dims}(state.volume.unchecked_get());

    Intersection result;
    Sense sense;
    SurfaceIndexer to_index(record_.surface_indexer_data);

    for (auto ax : range(Axis::size_))
    {
        auto dir = state.dir[to_int(ax)];

        // Ignore any stationary axis
        if (dir == 0)
        {
            continue;
        }

        auto target_coord = coords[to_int(ax)] + static_cast<int>(dir > 0);

        auto target_value = this->make_grid(ax)[target_coord];

        real_type dist
            = (target_value - static_cast<real_type>(state.pos[to_int(ax)]))
              / state.dir[to_int(ax)];

        if (dist > 0 && is_valid(dist) && dist < result.distance)
        {
            result.distance = dist;

            sense = dir > 0 ? Sense::inside : Sense::outside;
            auto local_surface = LocalSurfaceId(
                to_index({static_cast<size_type>(to_int(ax)), target_coord}));
            result.surface = detail::OnLocalSurface(local_surface, sense);
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Find the index of axis (x/y/z) we are about to cross:
 */
CELER_FUNCTION size_type
RectArrayTracker::find_surface_axis_idx(LocalSurfaceId s) const
{
    SurfaceInverseIndexer to_axis(record_.surface_indexer_data);
    return to_axis(s.unchecked_get())[0];
}

//---------------------------------------------------------------------------//
/*!
 * Create Grid object for a given axis.
 */
CELER_FUNCTION RectArrayTracker::Grid RectArrayTracker::make_grid(Axis ax) const
{
    return Grid(record_.grid[to_int(ax)], params_.reals);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
