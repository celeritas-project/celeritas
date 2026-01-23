//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitInserter.cc
//---------------------------------------------------------------------------//
#include "UnitInserter.hh"

#include <algorithm>
#include <array>
#include <iostream>
#include <regex>
#include <set>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/Ref.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/sys/Environment.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeTypes.hh"

#include "UniverseInserter.hh"
#include "../OrangeInput.hh"
#include "../surf/LocalSurfaceVisitor.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
constexpr int invalid_depth = -1;

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum CSG logic depth of a volume definition.
 *
 * Return a sentinel if the definition is invalid so that we can raise an
 * assertion in the caller with more context.
 */
int calc_depth(Span<logic_int const> logic)
{
    CELER_EXPECT(!logic.empty());

    // Calculate max depth
    int depth = 1;
    int cur_depth = 0;

    for (auto id : logic)
    {
        if (!logic::is_operator_token(id) || id == logic::ltrue)
        {
            ++cur_depth;
        }
        else if (id == logic::land || id == logic::lor)
        {
            depth = std::max(cur_depth, depth);
            --cur_depth;
        }
    }
    if (cur_depth != 1)
    {
        // Input definition is invalid; return a sentinel value
        depth = invalid_depth;
    }
    CELER_ENSURE(depth > 0 || depth == invalid_depth);
    return depth;
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
std::vector<Label> make_surface_labels(UnitInput& inp)
{
    CELER_EXPECT(inp.surface_labels.empty()
                 || inp.surface_labels.size() == inp.surfaces.size());

    std::vector<Label> result;
    result.resize(inp.surfaces.size());

    for (auto i : range(inp.surface_labels.size()))
    {
        Label surface_label = std::move(inp.surface_labels[i]);
        if (surface_label.ext.empty())
        {
            surface_label.ext = inp.label.name;
        }
        result[i] = std::move(surface_label);
    }
    inp.surface_labels.clear();
    return result;
}

//---------------------------------------------------------------------------//
//! Construct volume labels from the input volumes
auto make_volume_labels(UnitInput const& inp)
{
    UniverseInserter::VecVarLabel result;
    for (auto const& v : inp.volumes)
    {
        // Convert a <Label, VolInstId> -> <Label, VolInstId, VolId>
        // using a default extension
        result.emplace_back(
            std::visit(return_as<UniverseInserter::VariantLabel>(Overload{
                           [](auto&& obj) { return obj; },
                           [&inp](Label const& label) {
                               Label result = label;
                               // Add the unit's name as an extension if blank
                               if (result.ext.empty())
                               {
                                   result.ext = inp.label.name;
                               }
                               return result;
                           },
                       }),
                       v.label));
    }

    if (auto const& bg = inp.background)
    {
        CELER_ASSERT(bg.volume < result.size());
        result[bg.volume.get()] = bg.label;
    }

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a bounding box bumper for a given tolerance.
 *
 * This bumper will convert *to* fast real type *from* regular
 * real type. It conservatively expand to twice the potential bump distance
 * from a boundary so that the bbox will enclose the point even after a
 * potential bump.
 */
BoundingBoxBumper<fast_real_type, real_type> make_bumper(Tolerance<> const& tol)
{
    Tolerance<real_type> bbox_tol;
    bbox_tol.rel = 2 * tol.rel;
    bbox_tol.abs = 2 * tol.abs;
    CELER_ENSURE(bbox_tol);
    return BoundingBoxBumper<fast_real_type, real_type>(std::move(bbox_tol));
}

struct ForceMax
{
    size_type faces = std::numeric_limits<size_type>::max();
    size_type intersections = std::numeric_limits<size_type>::max();
};

static char const mfi_hack_envname[] = "ORANGE_MAX_FACE_INTERSECT";

//---------------------------------------------------------------------------//
/*!
 * Force maximum faces/intersections.
 *
 * This is if we know the "automatic" value is wrong, specifically if all
 * complicated/background cells are unreachable.
 *
 * See https://github.com/celeritas-project/celeritas/issues/1334 .
 */
ForceMax const& forced_scalar_max()
{
    static ForceMax const result = [] {
        std::string mfi = celeritas::getenv(mfi_hack_envname);
        if (mfi.empty())
        {
            return ForceMax{};
        }
        CELER_LOG(warning)
            << "Using a temporary, unsupported, and dangerous hack to "
               "override maximum faces and intersections in ORANGE: "
            << mfi_hack_envname << "='" << mfi << "'";

        ForceMax result;
        static std::regex const mfi_regex{R"re(^(\d+),(\d+)$)re"};
        std::smatch mfi_match;
        CELER_VALIDATE(std::regex_match(mfi, mfi_match, mfi_regex),
                       << "invalid pattern for " << mfi_hack_envname);
        auto get = [](char const* type, auto const& submatch) {
            auto&& s = submatch.str();
            auto updated = std::stoi(s);
            CELER_VALIDATE(updated > 0,
                           << "invalid maximum " << type << ": " << updated);
            CELER_LOG(warning)
                << "Forcing maximum " << type << " to " << updated;
            return static_cast<size_type>(updated);
        };
        result.faces = get("faces", mfi_match[0]);
        result.intersections = get("intersections", mfi_match[1]);
        return result;
    }();
    return result;
}

//---------------------------------------------------------------------------//
std::string to_string(VolumeInput::VariantLabel const& vlabel)
{
    return std::visit(Overload{[](Label const& lab) { return to_string(lab); },
                               [](VolumeInstanceId const& id) -> std::string {
                                   if (!id)
                                       return "<null>";
                                   return "vi " + std::to_string(id.get());
                               }},
                      vlabel);
}

//---------------------------------------------------------------------------//
/*!
 * Create a vector (indexed by local volume ID) of local canonical parents.
 *
 * This simply expands a sparse map into a full vector. The indices are all
 * local implementation volume IDs, even though the relationship they describe
 * is the "canonical" volume structure.
 */
std::vector<LocalVolumeId>
make_local_parent_vec(LocalVolumeId::size_type num_volumes,
                      UnitInput::MapLocalParent const& local_parent_map)
{
    CELER_EXPECT(num_volumes > 0);
    CELER_EXPECT(!local_parent_map.empty());

    std::vector<LocalVolumeId> local_parents(num_volumes);
    // Fill local parents
    for (auto&& [child, parent] : local_parent_map)
    {
        CELER_ASSERT(child < num_volumes);
        CELER_ASSERT(parent < num_volumes);
        local_parents[child.get()] = parent;
    }

    return local_parents;
}

//---------------------------------------------------------------------------//
/*!
 * Determine relative canonical volume levels of each local volume.
 *
 * Use a depth-first search to fill an array, indexed by local impl volumes, of
 * the volume relative to the top (most enclosing/closest to "world").
 */
std::vector<vol_level_uint>
make_local_level_vec(std::vector<LocalVolumeId> const& local_parents)
{
    constexpr vol_level_uint not_visited{static_cast<vol_level_uint>(-1)};
    std::vector<vol_level_uint> local_vol_level(local_parents.size(),
                                                not_visited);
    std::vector<LocalVolumeId> stack;

    // Traverse all local levels with DFS, excluding unreachable "exterior"
    // We loop over all volumes because we don't know a priori which one is the
    // "top" volume in the universe (could be background, could be explicit)
    for (auto lv_id : range(orange_exterior_volume + 1,
                            id_cast<LocalVolumeId>(local_parents.size())))
    {
        if (local_vol_level[lv_id.get()] != not_visited)
        {
            continue;
        }
        stack.push_back(lv_id);
        while (!stack.empty())
        {
            // Guard against cycles, which shouldn't be possible to construct
            CELER_ASSERT(stack.size() < VolumeLevelId{}.unchecked_get());

            auto child = stack.back();
            auto parent = local_parents[child.get()];
            vol_level_uint child_level{not_visited};
            if (parent)
            {
                child_level = local_vol_level[parent.get()];
                if (child_level == not_visited)
                {
                    // Parent has not yet been visited; go deeper
                    stack.push_back(parent);
                    continue;
                }
                else
                {
                    // Child is one deeper than parent
                    ++child_level;
                }
            }
            else
            {
                // No enclosing local volume: level zero
                child_level = 0;
            }

            // Save local level
            CELER_ASSERT(child_level != not_visited);
            local_vol_level[child.get()] = child_level;
            stack.pop_back();
        }
    }

    return local_vol_level;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from full parameter data.
 */
UnitInserter::UnitInserter(UniverseInserter* insert_universe, Data* orange_data)
    : orange_data_(orange_data)
    , build_bih_tree_{&orange_data_->bih_tree_data, BIHBuilder::Input{2}}
    , insert_transform_{&orange_data_->transforms, &orange_data_->reals}
    , build_surfaces_{&orange_data_->surface_types,
                      &orange_data_->real_ids,
                      &orange_data_->reals}
    , insert_universe_{insert_universe}
    , simple_units_{&orange_data_->simple_units}
    , local_surface_ids_{&orange_data_->local_surface_ids}
    , local_volume_ids_{&orange_data_->local_volume_ids}
    , real_ids_{&orange_data_->real_ids}
    , vl_uints_{&orange_data_->vl_uints}
    , logic_ints_{&orange_data_->logic_ints}
    , reals_{&orange_data_->reals}
    , surface_types_{&orange_data_->surface_types}
    , connectivity_records_{&orange_data_->connectivity_records}
    , volume_records_{&orange_data_->volume_records}
    , obz_records_{&orange_data_->obz_records}
    , daughters_{&orange_data_->daughters}
    , calc_bumped_(make_bumper(orange_data_->scalars.tol))
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
UnivId UnitInserter::operator()(UnitInput&& inp)
{
    CELER_VALIDATE(inp,
                   << "simple unit '" << inp.label
                   << "' is not properly constructed");

    SimpleUnitRecord unit;

    // Insert surfaces
    unit.surfaces = this->build_surfaces_(inp.surfaces);

    // Define volumes
    std::vector<VolumeRecord> vol_records(inp.volumes.size());
    std::vector<std::set<LocalVolumeId>> connectivity(inp.surfaces.size());
    std::vector<FastBBox> bboxes;
    BIHBuilder::SetLocalVolId implicit_vol_ids;
    for (auto i : range<LocalVolumeId::size_type>(inp.volumes.size()))
    {
        vol_records[i] = this->insert_volume(unit.surfaces, inp.volumes[i]);
        CELER_ASSERT(!vol_records.empty());

        // Store the bbox or an infinite bbox placeholder
        if (inp.volumes[i].bbox)
        {
            bboxes.push_back(calc_bumped_(inp.volumes[i].bbox));
        }
        else
        {
            bboxes.push_back(BoundingBox<fast_real_type>::from_infinite());
        }

        // Create a set of background volume ids for BIH construction
        if (inp.volumes[i].flags & VolumeRecord::Flags::implicit_vol)
        {
            implicit_vol_ids.insert(id_cast<LocalVolumeId>(i));
        }

        // Add oriented bounding zone record
        if (inp.volumes[i].obz)
        {
            this->process_obz_record(&(vol_records[i]), inp.volumes[i].obz);
        }

        // Add embedded universes
        if (inp.daughter_map.find(LocalVolumeId(i)) != inp.daughter_map.end())
        {
            this->process_daughter(&(vol_records[i]),
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

        CELER_VALIDATE(LocalVolumeId{i} == orange_exterior_volume
                           || inp.volumes[i].zorder != ZOrder::exterior,
                       << "only local volume 0 can be exterior");
    }

    // Save local parent IDs and local volume level
    if (!inp.local_parent_map.empty())
    {
        auto parents
            = make_local_parent_vec(inp.volumes.size(), inp.local_parent_map);
        auto levels = make_local_level_vec(parents);
        CELER_ASSERT(parents.size() == levels.size());

        unit.local_parent
            = local_volume_ids_.insert_back(parents.begin(), parents.end());
        unit.local_vol_level
            = vl_uints_.insert_back(levels.begin(), levels.end());
    }

    // Save volumes
    unit.volumes = ItemMap<LocalVolumeId, SimpleUnitRecord::VolumeRecordId>(
        volume_records_.insert_back(vol_records.begin(), vol_records.end()));

    // Create BIH tree
    {
        std::vector<size_type> invalid;
        for (auto i : range(bboxes.size()))
        {
            if (!bboxes[i] || is_half_inf(bboxes[i]))
            {
                invalid.push_back(i);
            }
        }
        CELER_VALIDATE(
            invalid.empty(),
            << "invalid (null or half-infinite) bounding boxes in '"
            << inp.label << "': "
            << join_stream(invalid.begin(),
                           invalid.end(),
                           ", ",
                           [&inp, &bboxes](std::ostream& os, size_type i) {
                               os << i << "='"
                                  << to_string(inp.volumes[i].label)
                                  << "': " << bboxes[i];
                           }));
    }
    unit.bih_tree = build_bih_tree_(std::move(bboxes), implicit_vol_ids);

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
        unit.background = id_cast<LocalVolumeId>(inp.volumes.size() - 1);
    }

    // Simple safety if all volumes provide support, excluding the external
    // volume, which appears first in the list
    static_assert(orange_exterior_volume == LocalVolumeId{0});
    unit.simple_safety = std::all_of(
        vol_records.begin() + 1, vol_records.end(), [](VolumeRecord const& v) {
            return supports_simple_safety(v.flags);
        });

    CELER_ASSERT(unit);
    simple_units_.push_back(unit);
    auto surf_labels = make_surface_labels(inp);
    auto vol_labels = make_volume_labels(inp);
    return (*insert_universe_)(UnivType::simple,
                               std::move(inp.label),
                               std::move(surf_labels),
                               std::move(vol_labels));
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

    // Mark this volume as "simple safety" if all of its constituent surfaces
    // support it, even in the case where the volume is implicit
    bool simple_safety = true;
    for (LocalSurfaceId sid : v.faces)
    {
        simple_safety = simple_safety
                        && visit_surface(SimpleSafetyGetter{}, sid);
    }

    // Calculate the max_intersection for the volume by summing up the
    // max_intersection for all constituent surfaces. If the volume is
    // background (implicit), no intersection is possible, thus
    // max_intersections is zero
    size_type max_intersections = 0;
    if (v.zorder != ZOrder::background)
    {
        for (LocalSurfaceId sid : v.faces)
        {
            max_intersections += visit_surface(NumIntersectionGetter{}, sid);
        }
    }

    static auto const nowhere_logic = []() -> std::array<logic_int, 2> {
        if (orange_tracking_logic() == LogicNotation::postfix)
            return {logic::ltrue, logic::lnot};
        else
            return {logic::lnot, logic::ltrue};
    }();

    auto input_logic = make_span(v.logic);
    if (v.zorder == ZOrder::background)
    {
        // "Background" volumes should not be explicitly reachable by logic or
        // BIH
        CELER_EXPECT(std::equal(input_logic.begin(),
                                input_logic.end(),
                                std::begin(nowhere_logic),
                                std::end(nowhere_logic)));
        CELER_EXPECT(!v.bbox);
        CELER_EXPECT(!v.obz);
        CELER_EXPECT(v.flags & VolumeRecord::implicit_vol);
        // Rely on incoming flags for "simple_safety": false from .org.json,
        // maybe true if built from GDML
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

    if (output.max_intersections > forced_scalar_max().intersections
        || output.faces.size() > forced_scalar_max().faces)
    {
        CELER_LOG(warning) << "Max intersections (" << output.max_intersections
                           << ") and/or faces (" << output.faces.size()
                           << ") exceed limits of '" << mfi_hack_envname
                           << " in volume '" << to_string(v.label)
                           << "': replacing with unreachable volume";

        output.faces = {};
        output.logic = logic_ints_.insert_back(std::begin(nowhere_logic),
                                               std::end(nowhere_logic));
        output.max_intersections = 0;
        output.flags = VolumeRecord::implicit_vol
                       | VolumeRecord::Flags::simple_safety;
    }

    // Calculate the maximum stack depth of the volume definition
    int depth = calc_depth(input_logic);
    CELER_VALIDATE(depth > 0,
                   << "invalid logic definition: operators do not balance");

    // Update global max faces/intersections/logic
    OrangeParamsScalars& scalars = orange_data_->scalars;
    inplace_max<size_type>(&scalars.max_faces, output.faces.size());
    inplace_max<size_type>(&scalars.max_intersections,
                           output.max_intersections);
    inplace_max<size_type>(&scalars.max_csg_levels, depth);

    return output;
}

//---------------------------------------------------------------------------//
/*!
 * Process a single oriented bounding zone record.
 */
void UnitInserter::process_obz_record(VolumeRecord* vol_record,
                                      OrientedBoundingZoneInput const& obz_input)
{
    CELER_EXPECT(obz_input);

    OrientedBoundingZoneRecord obz_record;

    // Set half widths
    auto inner_hw = calc_half_widths(calc_bumped_(obz_input.inner));
    auto outer_hw = calc_half_widths(calc_bumped_(obz_input.outer));
    obz_record.half_widths = {inner_hw, outer_hw};

    // Set offsets
    auto inner_offset_id = insert_transform_(VariantTransform{
        std::in_place_type<Translation>, calc_center(obz_input.inner)});
    auto outer_offset_id = insert_transform_(VariantTransform{
        std::in_place_type<Translation>, calc_center(obz_input.outer)});
    obz_record.offset_ids = {inner_offset_id, outer_offset_id};

    // Set transformation
    obz_record.trans_id = obz_input.trans_id;

    // Save the OBZ record to the volume record
    vol_record->obz_id = obz_records_.push_back(obz_record);
}

//---------------------------------------------------------------------------//
/*!
 * Process a single daughter universe.
 */
void UnitInserter::process_daughter(VolumeRecord* vol_record,
                                    DaughterInput const& daughter_input)
{
    Daughter daughter;
    daughter.univ_id = daughter_input.univ_id;
    daughter.trans_id = insert_transform_(daughter_input.transform);

    vol_record->daughter_id = daughters_.push_back(daughter);
    vol_record->flags |= VolumeRecord::embedded_universe;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
