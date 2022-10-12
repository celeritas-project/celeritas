//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitInserter.cc
//---------------------------------------------------------------------------//
#include "UnitInserter.hh"

#include <algorithm>
#include <set>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/surf/SurfaceAction.hh"
#include "orange/surf/Surfaces.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum logic depth of a volume definition.
 *
 * Return 0 if the definition is invalid so that we can raise an assertion in
 * the caller with more context.
 */
int calc_max_depth(Span<const logic_int> logic)
{
    CELER_EXPECT(!logic.empty());
    // Calculate max depth
    int max_depth = 1;
    int cur_depth = 0;

    for (auto id : logic)
    {
        if (!logic::is_operator_token(id) || id == logic::ltrue)
        {
            ++cur_depth;
        }
        else if (id == logic::land || id == logic::lor)
        {
            max_depth = std::max(cur_depth, max_depth);
            --cur_depth;
        }
    }
    if (cur_depth != 1)
    {
        // Input definition is invalid; return a sentinel value
        max_depth = 0;
    }
    return max_depth;
}

//---------------------------------------------------------------------------//
/*!
 * Whether a volume supports "simple safety".
 *
 * We declare this to be true for "implicit" cells (whose interiors aren't
 * tracked like normal cells), as well as cells that have *both* the simple
 * safety flag (no invalid surface types) *and* no internal surfaces.
 */
bool supports_simple_safety(logic_int flags)
{
    return (flags & VolumeRecord::implicit_cell)
           || ((flags & VolumeRecord::simple_safety)
               && !(flags & VolumeRecord::internal_surfaces));
}

//---------------------------------------------------------------------------//
//! More readable `X = max(X, Y)` with same semantics as atomic_max
template<class T>
T inplace_max(T* target, T val)
{
    T orig  = *target;
    *target = celeritas::max(orig, val);
    return orig;
}

//---------------------------------------------------------------------------//
//! Static surface action for getting the storage requirements for a surface.
template<class T>
struct SurfaceDataSize
{
    constexpr size_type operator()() const noexcept
    {
        return T::Storage::extent;
    }
};

//---------------------------------------------------------------------------//
//! Return a surface's "simple" flag
struct SimpleSafetyGetter
{
    template<class S>
    constexpr bool operator()(const S&) const noexcept
    {
        return S::simple_safety();
    }
};

//---------------------------------------------------------------------------//
//! Return the number of intersections for a surface
struct NumIntersectionGetter
{
    template<class S>
    constexpr size_type operator()(const S&) const noexcept
    {
        using Intersections = typename S::Intersections;
        return Intersections{}.size();
    }
};

//---------------------------------------------------------------------------//
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from full parameter data.
 */
UnitInserter::UnitInserter(Data* orange_data) : orange_data_(orange_data)
{
    CELER_EXPECT(orange_data);

    // Initialize scalars
    orange_data_->scalars.max_faces         = 1;
    orange_data_->scalars.max_intersections = 1;
}

//---------------------------------------------------------------------------//
/*!
 * Create a simple unit and return its ID.
 */
SimpleUnitId UnitInserter::operator()(const UnitInput& inp)
{
    SimpleUnitRecord unit;

    // Insert surfaces
    unit.surfaces = this->insert_surfaces(inp.surfaces);

    // Define volumes
    std::vector<VolumeRecord>       vol_records(inp.volumes.size());
    std::vector<std::set<VolumeId>> connectivity(inp.surfaces.size());
    for (auto i : range(inp.volumes.size()))
    {
        vol_records[i] = this->insert_volume(unit.surfaces, inp.volumes[i]);
        CELER_ASSERT(!vol_records.empty());

        // Add connectivity
        for (SurfaceId f : inp.volumes[i].faces)
        {
            CELER_ASSERT(f < connectivity.size());
            connectivity[f.unchecked_get()].insert(VolumeId(i));
        }
    }

    // Save volumes
    unit.volumes = make_builder(&orange_data_->volume_records)
                       .insert_back(vol_records.begin(), vol_records.end());

    // Save connectivity
    {
        std::vector<Connectivity> conn(connectivity.size());
        CELER_ASSERT(conn.size() == unit.surfaces.types.size());
        auto vol_ids = make_builder(&orange_data_->volume_ids);
        for (auto i : range(connectivity.size()))
        {
            Connectivity c;
            c.neighbors = vol_ids.insert_back(connectivity[i].begin(),
                                              connectivity[i].end());
            conn[i]     = c;
        }
        unit.connectivity = make_builder(&orange_data_->connectivities)
                                .insert_back(conn.begin(), conn.end());
    }

    // Save unit scalars
    unit.simple_safety = std::all_of(
        vol_records.begin(), vol_records.end(), [](const VolumeRecord& v) {
            return supports_simple_safety(v.flags);
        });

    CELER_ASSERT(unit);
    auto simple_units = make_builder(&orange_data_->simple_unit);
    return simple_units.push_back(unit);
}

//---------------------------------------------------------------------------//
/*!
 * Insert all surfaces at once.
 */
SurfacesRecord UnitInserter::insert_surfaces(const SurfaceInput& s)
{
    using RealId = SurfacesRecord::RealId;

    //// Check input consistency ////

    CELER_VALIDATE(s.types.size() == s.sizes.size(),
                   << "inconsistent surfaces input: number of types ("
                   << s.types.size() << ") must match number of sizes ("
                   << s.sizes.size() << ")");

    auto get_data_size = make_static_surface_action<SurfaceDataSize>();

    size_type accum_size = 0;
    for (auto i : range(s.types.size()))
    {
        size_type expected_size = get_data_size(s.types[i]);
        CELER_VALIDATE(expected_size == s.sizes[i],
                       << "inconsistent surface data size (" << s.sizes[i]
                       << ") for entry " << i << ": "
                       << "surface type " << to_cstring(s.types[i])
                       << " should have " << expected_size);
        accum_size += expected_size;
    }

    CELER_VALIDATE(accum_size == s.data.size(),
                   << "incorrect surface data size (" << s.data.size()
                   << "): should match accumulated sizes (" << accum_size
                   << ")");

    //// Insert data ////

    // Insert surface types
    SurfacesRecord result;
    auto           types = make_builder(&orange_data_->surface_types);
    result.types         = types.insert_back(s.types.begin(), s.types.end());

    // Insert surface data all at once
    auto reals      = make_builder(&orange_data_->reals);
    auto real_range = reals.insert_back(s.data.begin(), s.data.end());

    RealId           next_offset = real_range.front();
    auto             offsets     = make_builder(&orange_data_->real_ids);
    OpaqueId<RealId> start_offset(offsets.size());
    offsets.reserve(offsets.size() + s.sizes.size());
    for (auto single_size : s.sizes)
    {
        offsets.push_back(next_offset);
        next_offset = next_offset + single_size;
    }
    CELER_ASSERT(next_offset == *real_range.end());

    result.data_offsets = range(start_offset, start_offset + s.sizes.size());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Insert data from a single volume.
 */
VolumeRecord UnitInserter::insert_volume(const SurfacesRecord& surf_record,
                                         const VolumeInput&    v)
{
    CELER_EXPECT(v);
    CELER_EXPECT(std::is_sorted(v.faces.begin(), v.faces.end()));
    CELER_EXPECT(v.faces.empty() || v.faces.back() < surf_record.types.size());

    auto     params_cref = make_const_ref(*orange_data_);
    Surfaces surfaces{params_cref, surf_record};

    // Mark as 'simple safety' if all the surfaces are simple
    bool      simple_safety     = true;
    logic_int max_intersections = 0;

    auto get_simple_safety
        = make_surface_action(surfaces, SimpleSafetyGetter{});
    auto get_num_intersections
        = make_surface_action(surfaces, NumIntersectionGetter{});

    for (SurfaceId sid : v.faces)
    {
        CELER_ASSERT(sid < surfaces.num_surfaces());
        simple_safety = simple_safety && get_simple_safety(sid);
        max_intersections += get_num_intersections(sid);
    }

    auto faces = make_builder(&orange_data_->surface_ids);
    auto logic = make_builder(&orange_data_->logic_ints);

    VolumeRecord output;
    output.faces = faces.insert_back(v.faces.begin(), v.faces.end());
    output.logic = logic.insert_back(v.logic.begin(), v.logic.end());
    output.max_intersections = max_intersections;
    output.flags             = v.flags;
    if (simple_safety)
    {
        output.flags |= VolumeRecord::Flags::simple_safety;
    }

    // Calculate the maximum stack depth of the volume definition
    int max_depth = calc_max_depth(make_span(v.logic));
    CELER_VALIDATE(max_depth > 0,
                   << "invalid logic definition: operators do not balance");

    // Update global max faces/intersections/logic
    OrangeParamsScalars& scalars = orange_data_->scalars;
    inplace_max<size_type>(&scalars.max_faces, output.faces.size());
    inplace_max<size_type>(&scalars.max_intersections,
                           output.max_intersections);
    inplace_max<size_type>(&scalars.max_logic_depth, max_depth);

    return output;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
