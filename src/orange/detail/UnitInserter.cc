//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitInserter.cc
//---------------------------------------------------------------------------//
#include "UnitInserter.hh"

#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/Ref.hh"
#include "corecel/math/Algorithms.hh"
#include "orange/BoundingBoxUtils.hh"
#include "orange/construct/OrangeInput.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
constexpr int invalid_max_depth = -1;

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum logic depth of a volume definition.
 *
 * Return 0 if the definition is invalid so that we can raise an assertion in
 * the caller with more context.
 */
int calc_max_depth(Span<logic_int const> logic)
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
        max_depth = invalid_max_depth;
    }
    CELER_ENSURE(max_depth > 0 || max_depth == invalid_max_depth);
    return max_depth;
}

//---------------------------------------------------------------------------//
/*!
 * Whether a volume supports "simple safety".
 *
 * We declare this to be true for "implicit" volumes (whose interiors aren't
 * tracked like normal volumes), as well as volumes that have *both* the simple
 * safety flag (no invalid surface types) *and* no internal surfaces.
 */
bool supports_simple_safety(logic_int flags)
{
    return (flags & VolumeRecord::implicit_vol)
           || ((flags & VolumeRecord::simple_safety)
               && !(flags & VolumeRecord::internal_surfaces));
}

//---------------------------------------------------------------------------//
//! More readable `X = max(X, Y)` with same semantics as atomic_max
template<class T>
T inplace_max(T* target, T val)
{
    T orig = *target;
    *target = celeritas::max(orig, val);
    return orig;
}

//---------------------------------------------------------------------------//
//! Return a surface's "simple" flag
struct SimpleSafetyGetter
{
    template<class S>
    constexpr bool operator()(S const&) const noexcept
    {
        return S::simple_safety();
    }
};

//---------------------------------------------------------------------------//
//! Return the number of intersections for a surface
struct NumIntersectionGetter
{
    template<class S>
    constexpr size_type operator()(S const&) const noexcept
    {
        using Intersections = typename S::Intersections;
        return Intersections{}.size();
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from full parameter data.
 */
UnitInserter::UnitInserter(Data* orange_data)
    : orange_data_(orange_data)
    , build_bih_tree_{&orange_data_->bih_tree_data}
    , insert_transform_{&orange_data_->transforms, &orange_data_->reals}
{
    CELER_EXPECT(orange_data);

    // Initialize scalars
    orange_data_->scalars.max_faces = 1;
    orange_data_->scalars.max_intersections = 1;
}

//---------------------------------------------------------------------------//
/*!
 * Create a simple unit and return its ID.
 */
SimpleUnitId UnitInserter::operator()(UnitInput const& inp)
{
    SimpleUnitRecord unit;

    // Insert surfaces
    unit.surfaces = this->insert_surfaces(inp.surfaces);

    // Define volumes
    std::vector<VolumeRecord> vol_records(inp.volumes.size());
    std::vector<std::set<LocalVolumeId>> connectivity(inp.surfaces.size());
    std::vector<FastBBox> bboxes;
    for (auto i : range(inp.volumes.size()))
    {
        vol_records[i] = this->insert_volume(unit.surfaces, inp.volumes[i]);
        CELER_ASSERT(!vol_records.empty());

        // Store the bbox or an infinite bbox placeholder
        if (inp.volumes[i].bbox)
        {
            bboxes.push_back(calc_bumped<fast_real_type>(inp.volumes[i].bbox));
        }
        else
        {
            bboxes.push_back(BoundingBox<fast_real_type>::from_infinite());
        }

        // Add embedded universes
        if (inp.daughter_map.find(LocalVolumeId(i)) != inp.daughter_map.end())
        {
            process_daughter(&(vol_records[i]),
                             inp.daughter_map.at(LocalVolumeId(i)));
        }

        // Add connectivity for explicitly connected volumes
        if (!(vol_records[i].flags & VolumeRecord::implicit_vol))
        {
            for (LocalSurfaceId f : inp.volumes[i].faces)
            {
                CELER_ASSERT(f < connectivity.size());
                connectivity[f.unchecked_get()].insert(LocalVolumeId(i));
            }
        }
    }

    // Save volumes
    unit.volumes = ItemMap<LocalVolumeId, SimpleUnitRecord::VolumeRecordId>(
        make_builder(&orange_data_->volume_records)
            .insert_back(vol_records.begin(), vol_records.end()));

    // Create BIH tree
    CELER_VALIDATE(std::all_of(bboxes.begin(),
                               bboxes.end(),
                               [](FastBBox const& b) { return b; }),
                   << "not all bounding boxes have been assigned");
    unit.bih_tree = build_bih_tree_(std::move(bboxes));

    // Save connectivity
    {
        std::vector<Connectivity> conn(connectivity.size());
        CELER_ASSERT(conn.size() == unit.surfaces.types.size());
        auto vol_ids = make_builder(&orange_data_->local_volume_ids);
        for (auto i : range(connectivity.size()))
        {
            Connectivity c;
            c.neighbors = vol_ids.insert_back(connectivity[i].begin(),
                                              connectivity[i].end());
            conn[i] = c;
        }
        unit.connectivity = make_builder(&orange_data_->connectivities)
                                .insert_back(conn.begin(), conn.end());
    }

    // Save unit scalars
    if (inp.volumes.back().zorder == 1)
    {
        unit.background = LocalVolumeId(inp.volumes.size() - 1);
    }
    unit.simple_safety = std::all_of(
        vol_records.begin(), vol_records.end(), [](VolumeRecord const& v) {
            return supports_simple_safety(v.flags);
        });

    CELER_ASSERT(unit);
    return make_builder(&orange_data_->simple_units).push_back(unit);
}

//---------------------------------------------------------------------------//
/*!
 * Insert all surfaces at once.
 */
SurfacesRecord UnitInserter::insert_surfaces(SurfaceInput const& s)
{
    using RealId = SurfacesRecord::RealId;

    //// Check input consistency ////

    CELER_VALIDATE(s.types.size() == s.sizes.size(),
                   << "inconsistent surfaces input: number of types ("
                   << s.types.size() << ") must match number of sizes ("
                   << s.sizes.size() << ")");

    auto get_data_size = [](auto surf_traits) {
        using Surface = typename decltype(surf_traits)::type;
        return Surface::Storage::extent;
    };

    size_type accum_size = 0;
    for (auto i : range(s.types.size()))
    {
        size_type expected_size = visit_surface_type(get_data_size, s.types[i]);
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
    auto types = make_builder(&orange_data_->surface_types);
    result.types = types.insert_back(s.types.begin(), s.types.end());

    // Insert surface data all at once
    auto reals = make_builder(&orange_data_->reals);
    auto real_range = reals.insert_back(s.data.begin(), s.data.end());

    RealId next_offset = real_range.front();
    auto offsets = make_builder(&orange_data_->real_ids);
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
VolumeRecord UnitInserter::insert_volume(SurfacesRecord const& surf_record,
                                         VolumeInput const& v)
{
    CELER_EXPECT(v);
    CELER_EXPECT(std::is_sorted(v.faces.begin(), v.faces.end()));
    CELER_EXPECT(v.faces.empty() || v.faces.back() < surf_record.types.size());

    auto params_cref = make_const_ref(*orange_data_);
    LocalSurfaceVisitor visit_surface(params_cref, surf_record);

    // Mark as 'simple safety' if all the surfaces are simple
    bool simple_safety = true;
    logic_int max_intersections = 0;

    for (LocalSurfaceId sid : v.faces)
    {
        simple_safety = simple_safety
                        && visit_surface(SimpleSafetyGetter{}, sid);
        max_intersections += visit_surface(NumIntersectionGetter{}, sid);
    }

    auto input_logic = make_span(v.logic);
    if (v.zorder == 1)
    {
        // Currently SCALE ORANGE writes background volumes as having "empty"
        // logic, whereas we really want them to be "nowhere" (at least
        // nowhere *explicitly* using the 'inside' logic). It gets away with
        // this because it always uses BVH to initialize, and the implicit
        // volumes get an empty bbox. To avoid special cases in Celeritas, set
        // the logic to be explicitly "not true".
        CELER_EXPECT(input_logic.empty());
        static const logic_int nowhere_logic[] = {logic::ltrue, logic::lnot};
        input_logic = make_span(nowhere_logic);
    }

    VolumeRecord output;
    output.faces = make_builder(&orange_data_->local_surface_ids)
                       .insert_back(v.faces.begin(), v.faces.end());
    output.logic = make_builder(&orange_data_->logic_ints)
                       .insert_back(input_logic.begin(), input_logic.end());
    output.max_intersections = max_intersections;
    output.flags = v.flags;
    if (simple_safety)
    {
        output.flags |= VolumeRecord::Flags::simple_safety;
    }

    // Calculate the maximum stack depth of the volume definition
    int max_depth = calc_max_depth(input_logic);
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
/*!
 * Process a single daughter universe.
 */
void UnitInserter::process_daughter(VolumeRecord* vol_record,
                                    DaughterInput const& daughter_input)
{
    Daughter daughter;
    daughter.universe_id = daughter_input.universe_id;
    daughter.transform_id = insert_transform_(daughter_input.translation);

    vol_record->daughter_id
        = make_builder(&orange_data_->daughters).push_back(daughter);
    vol_record->flags &= VolumeRecord::embedded_universe;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
