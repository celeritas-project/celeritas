//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
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
#include "corecel/data/Ref.hh"
#include "corecel/math/Algorithms.hh"

#include "UniverseInserter.hh"
#include "../BoundingBoxUtils.hh"
#include "../OrangeInput.hh"
#include "../surf/LocalSurfaceVisitor.hh"

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
//! Construct surface labels, empty if needed
std::vector<Label> make_surface_labels(UnitInput const& inp)
{
    CELER_EXPECT(inp.surface_labels.empty()
                 || inp.surface_labels.size() == inp.surfaces.size());

    std::vector<Label> result;
    result.resize(inp.surfaces.size());

    for (auto i : range(inp.surface_labels.size()))
    {
        Label surface_label = inp.surface_labels[i];
        if (surface_label.ext.empty())
        {
            surface_label.ext = inp.label.name;
        }
        result[i] = std::move(surface_label);
    }
    return result;
}

//---------------------------------------------------------------------------//
//! Construct volume labels from the input volumes
std::vector<Label> make_volume_labels(UnitInput const& inp)
{
    std::vector<Label> result;
    for (auto const& v : inp.volumes)
    {
        Label vl = v.label;
        if (vl.ext.empty())
        {
            vl.ext = inp.label.name;
        }
        result.push_back(std::move(vl));
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from full parameter data.
 */
UnitInserter::UnitInserter(UniverseInserter* insert_universe, Data* orange_data)
    : orange_data_(orange_data)
    , build_bih_tree_{&orange_data_->bih_tree_data}
    , insert_transform_{&orange_data_->transforms, &orange_data_->reals}
    , build_surfaces_{&orange_data_->surface_types,
                      &orange_data_->real_ids,
                      &orange_data_->reals}
    , insert_universe_{insert_universe}
    , simple_units_{&orange_data_->simple_units}
    , local_surface_ids_{&orange_data_->local_surface_ids}
    , local_volume_ids_{&orange_data_->local_volume_ids}
    , real_ids_{&orange_data_->real_ids}
    , logic_ints_{&orange_data_->logic_ints}
    , reals_{&orange_data_->reals}
    , surface_types_{&orange_data_->surface_types}
    , connectivity_records_{&orange_data_->connectivity_records}
    , volume_records_{&orange_data_->volume_records}
    , daughters_{&orange_data_->daughters}
{
    CELER_EXPECT(orange_data);
    CELER_EXPECT(orange_data->scalars.tol);

    // Initialize scalars
    orange_data_->scalars.max_faces = 1;
    orange_data_->scalars.max_intersections = 1;
}

//---------------------------------------------------------------------------//
/*!
 * Create a simple unit and return its ID.
 */
UniverseId UnitInserter::operator()(UnitInput const& inp)
{
    CELER_VALIDATE(inp,
                   << "simple unit '" << inp.label
                   << "' is not properly constructed");

    SimpleUnitRecord unit;

    // Insert surfaces
    unit.surfaces = this->build_surfaces_(inp.surfaces);

    // Bounding box bumper and converter *to* fast real type *from* regular
    // real type: conservatively expand to twice the potential bump distance
    // from a boundary so that the bbox will enclose the point even after a
    // potential bump
    BoundingBoxBumper<fast_real_type, real_type> calc_bumped{
        [&tol = orange_data_->scalars.tol] {
            Tolerance<real_type> bbox_tol;
            bbox_tol.rel = 2 * tol.rel;
            bbox_tol.abs = 2 * tol.abs;
            CELER_ENSURE(bbox_tol);
            return bbox_tol;
        }()};

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
            bboxes.push_back(calc_bumped(inp.volumes[i].bbox));
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
        volume_records_.insert_back(vol_records.begin(), vol_records.end()));

    // Create BIH tree
    CELER_VALIDATE(std::all_of(bboxes.begin(),
                               bboxes.end(),
                               [](FastBBox const& b) { return b; }),
                   << "not all bounding boxes have been assigned");
    unit.bih_tree = build_bih_tree_(std::move(bboxes));

    // Save connectivity
    {
        std::vector<ConnectivityRecord> conn(connectivity.size());
        for (auto i : range(connectivity.size()))
        {
            conn[i].neighbors = local_volume_ids_.insert_back(
                connectivity[i].begin(), connectivity[i].end());
        }
        unit.connectivity
            = connectivity_records_.insert_back(conn.begin(), conn.end());
    }

    // Save unit scalars
    if (inp.volumes.back().zorder == ZOrder::background)
    {
        unit.background = LocalVolumeId(inp.volumes.size() - 1);
    }
    unit.simple_safety = std::all_of(
        vol_records.begin(), vol_records.end(), [](VolumeRecord const& v) {
            return supports_simple_safety(v.flags);
        });

    CELER_ASSERT(unit);
    simple_units_.push_back(unit);
    return (*insert_universe_)(UniverseType::simple,
                               inp.label,
                               make_surface_labels(inp),
                               make_volume_labels(inp));
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
    size_type max_intersections = 0;

    for (LocalSurfaceId sid : v.faces)
    {
        simple_safety = simple_safety
                        && visit_surface(SimpleSafetyGetter{}, sid);
        max_intersections += visit_surface(NumIntersectionGetter{}, sid);
    }

    auto input_logic = make_span(v.logic);
    if (v.zorder == ZOrder::background)
    {
        // "Background" volumes should not be explicitly reachable by logic or
        // BIH
        static logic_int const nowhere_logic[] = {logic::ltrue, logic::lnot};
        CELER_EXPECT(std::equal(input_logic.begin(),
                                input_logic.end(),
                                std::begin(nowhere_logic),
                                std::end(nowhere_logic)));
        CELER_EXPECT(is_infinite(v.bbox));
    }

    VolumeRecord output;
    output.faces
        = local_surface_ids_.insert_back(v.faces.begin(), v.faces.end());
    output.logic
        = logic_ints_.insert_back(input_logic.begin(), input_logic.end());
    output.max_intersections = static_cast<logic_int>(max_intersections);
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
    daughter.transform_id = insert_transform_(daughter_input.transform);

    vol_record->daughter_id = daughters_.push_back(daughter);
    vol_record->flags &= VolumeRecord::embedded_universe;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
