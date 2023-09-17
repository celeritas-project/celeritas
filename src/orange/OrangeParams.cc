//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.cc
//---------------------------------------------------------------------------//
#include "OrangeParams.hh"

#include <fstream>
#include <initializer_list>
#include <numeric>
#include <utility>
#include <vector>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "orange/BoundingBox.hh"

#include "OrangeData.hh"  // IWYU pragma: associated
#include "OrangeTypes.hh"
#include "construct/OrangeInput.hh"
#include "detail/RectArrayInserter.hh"
#include "detail/UnitInserter.hh"
#include "univ/detail/LogicStack.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "construct/OrangeInputIO.json.hh"
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Load a geometry from the given filename.
 */
OrangeInput input_from_json(std::string filename)
{
    CELER_VALIDATE(CELERITAS_USE_JSON,
                   << "JSON is not enabled so geometry cannot be loaded");

    CELER_LOG(info) << "Loading ORANGE geometry from JSON at " << filename;
    ScopedTimeLog scoped_time;

    if (ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Using ORANGE geometry with GDML suffix: trying "
                              "`.org.json` instead";
        filename.erase(filename.end() - 5, filename.end());
        filename += ".org.json";
    }
    else if (!ends_with(filename, ".json"))
    {
        CELER_LOG(warning) << "Expected '.json' extension for JSON input";
    }

    OrangeInput result;

#if CELERITAS_USE_JSON
    std::ifstream infile(filename);
    CELER_VALIDATE(infile,
                   << "failed to open geometry at '" << filename << '\'');
    // Use the `from_json` defined in OrangeInputIO.json to read the JSON input
    nlohmann::json::parse(infile).get_to(result);
#endif

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate the maximum num of universe levels within the geometry
 *
 * A single-universe geometry will have a max_depth of 1.
 * The operator() is a recursive function, so for external callers, uid should
 * be the root universe id.
 */
struct MaxDepthCalculator
{
    HostVal<OrangeParamsData> const& data_;
    std::map<UniverseId, size_type> depths;

    // Construct from host data
    MaxDepthCalculator(HostVal<OrangeParamsData> const& data) : data_(data) {}

    // Calculate max depth
    size_type operator()(UniverseId const& uid)
    {
        CELER_EXPECT(uid);

        // Check for cached value
        auto&& [iter, inserted] = depths.insert({uid, {}});
        if (!inserted)
        {
            // Return cached value
            return iter->second;
        }

        // Depth of the deepest daughter of uid
        size_type max_sub_depth = 0;

        if (data_.universe_types[uid] == UniverseType::simple)
        {
            auto const& simple_unit_record
                = data_.simple_units[SimpleUnitId{data_.universe_indices[uid]}];
            for (auto vol_id : range(simple_unit_record.volumes.size()))
            {
                auto vol_record
                    = data_.volume_records[simple_unit_record
                                               .volumes[LocalVolumeId{vol_id}]];
                if (vol_record.daughter_id)
                {
                    max_sub_depth = std::max(
                        max_sub_depth,
                        (*this)(data_.daughters[vol_record.daughter_id]
                                    .universe_id));
                }
            }
        }
        else if (data_.universe_types[uid] == UniverseType::rect_array)
        {
            auto const& rect_array_record
                = data_.rect_arrays[RectArrayId{data_.universe_indices[uid]}];
            for (auto vol_id : range(rect_array_record.daughters.size()))
            {
                max_sub_depth = std::max(
                    max_sub_depth,
                    (*this)(data_
                                .daughters[rect_array_record
                                               .daughters[LocalVolumeId{vol_id}]]
                                .universe_id));
            }
        }
        else
        {
            CELER_ASSERT_UNREACHABLE();
        }

        // Depth of the deepest daughter, plus 1 for the current uid
        auto max_depth = max_sub_depth + 1;

        // Save to cache
        iter->second = max_depth;

        return max_depth;
    }
};
//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a JSON file (if JSON is enabled).
 *
 * The JSON format is defined by the SCALE ORANGE exporter (not currently
 * distributed).
 */
OrangeParams::OrangeParams(std::string const& json_filename)
    : OrangeParams(input_from_json(json_filename))
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct in-memory from a Geant4 geometry (not implemented).
 *
 * Perhaps someday we'll implement in-memory translation...
 */
OrangeParams::OrangeParams(G4VPhysicalVolume const*)
{
    CELER_NOT_IMPLEMENTED("Geant4->VecGeom geometry translation");
}

//---------------------------------------------------------------------------//
/*!
 * Advanced usage: construct from explicit host data.
 *
 * Volume and surface labels must be unique for the time being.
 */
OrangeParams::OrangeParams(OrangeInput input)
{
    CELER_VALIDATE(input, << "input geometry is incomplete");
    CELER_VALIDATE(!input.universes.empty(),
                   << "input geometry has no universes");

    if (!input.tol)
    {
        input.tol = Tolerance<>::from_default();
    }

    // Create host data for construction, setting tolerances first
    HostVal<OrangeParamsData> host_data;
    host_data.scalars.tol = input.tol;

    // Calculate offsets for UniverseIndexerData
    {
        auto ui_surf = make_builder(&host_data.universe_indexer_data.surfaces);
        auto ui_vol = make_builder(&host_data.universe_indexer_data.volumes);
        ui_surf.push_back(0);
        ui_vol.push_back(0);

        auto get_num_surfaces = Overload{
            [](UnitInput const& u) -> size_type { return u.surfaces.size(); },
            [](RectArrayInput const& r) -> size_type {
                return std::accumulate(r.grid.begin(),
                                       r.grid.end(),
                                       size_type(0),
                                       [](size_type acc, const auto& vec) {
                                           return acc + vec.size();
                                       });
            }};

        auto get_num_volumes = Overload{
            [](UnitInput const& u) -> size_type { return u.volumes.size(); },
            [](RectArrayInput const& r) -> size_type {
                return r.daughters.size();
            }};

        for (auto const& u : input.universes)
        {
            using AllVals = AllItems<size_type, MemSpace::native>;

            auto surface_offset
                = host_data.universe_indexer_data.surfaces[AllVals{}].back();
            auto volume_offset
                = host_data.universe_indexer_data.volumes[AllVals{}].back();

            auto num_surfs = std::visit(get_num_surfaces, u);
            auto num_vols = std::visit(get_num_volumes, u);

            ui_surf.push_back(surface_offset + num_surfs);
            ui_vol.push_back(volume_offset + num_vols);
        }
    }

    // Create universe_types and universe_indices vectors
    {
        auto u_types_builder = make_builder(&host_data.universe_types);
        auto u_indices_builder = make_builder(&host_data.universe_indices);

        std::vector<size_type> current_indices(
            static_cast<size_t>(UniverseType::size_), 0);

        for (auto const& u : input.universes)
        {
            auto u_type_idx = u.index();
            u_types_builder.push_back(static_cast<UniverseType>(u_type_idx));
            u_indices_builder.push_back(current_indices[u_type_idx]++);
        }
    }

    // Insert all universes
    {
        detail::UnitInserter insert_unit(&host_data);
        detail::RectArrayInserter insert_rect_array(&host_data);

        auto insert_universe = Overload{
            [&insert_unit](UnitInput const& u) {
                CELER_VALIDATE(u,
                               << "simple unit '" << u.label
                               << "' is not properly constructed");

                insert_unit(u);
            },
            [&insert_rect_array](RectArrayInput const& r) {
                CELER_VALIDATE(r,
                               << "rect array '" << r.label
                               << "' is not properly constructed");

                insert_rect_array(r);
            },
        };

        for (auto const& u : input.universes)
        {
            std::visit(insert_universe, u);
        }
    }

    // Get surface/volume labels
    this->process_metadata(input);

    supports_safety_ = host_data.simple_units[SimpleUnitId{0}].simple_safety;

    CELER_VALIDATE(std::holds_alternative<UnitInput>(input.universes.front()),
                   << "global universe is not a SimpleUnit");
    bbox_ = std::get<UnitInput>(input.universes.front()).bbox;

    // Update scalars *after* loading all units
    host_data.scalars.max_depth = MaxDepthCalculator{host_data}(UniverseId{0});
    CELER_VALIDATE(host_data.scalars.max_logic_depth
                       < detail::LogicStack::max_stack_depth(),
                   << "input geometry has at least one volume with a "
                      "logic depth of"
                   << host_data.scalars.max_logic_depth
                   << " (surfaces are nested too deeply); but the logic "
                      "stack is limited to a depth of "
                   << detail::LogicStack::max_stack_depth());

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};

    CELER_ENSURE(data_);
    CELER_ENSURE(vol_labels_.size() > 0);
    CELER_ENSURE(bbox_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a volume.
 */
Label const& OrangeParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_.get(vol);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a unique name.
 *
 * If the name isn't in the geometry, a null ID will be returned. If the name
 * is not unique, a RuntimeError will be raised.
 */
VolumeId OrangeParams::find_volume(std::string const& name) const
{
    auto result = vol_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "volume '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a label.
 *
 * If the label isn't in the geometry, a null ID will be returned.
 */
VolumeId OrangeParams::find_volume(Label const& label) const
{
    return vol_labels_.find(label);
}

//---------------------------------------------------------------------------//
/*!
 * Get zero or more volume IDs corresponding to a name.
 *
 * This is useful for volumes that are repeated in the geometry with different
 * uniquifying 'extensions'.
 */
auto OrangeParams::find_volumes(std::string const& name) const
    -> SpanConstVolumeId
{
    return vol_labels_.find_all(name);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a surface.
 */
Label const& OrangeParams::id_to_label(SurfaceId surf) const
{
    CELER_EXPECT(surf < surf_labels_.size());
    return surf_labels_.get(surf);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the surface ID corresponding to a label name.
 */
SurfaceId OrangeParams::find_surface(std::string const& name) const
{
    auto result = surf_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "surface '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get surface and volume labels for all universes.
 */
void OrangeParams::process_metadata(OrangeInput const& input)
{
    std::vector<Label> surface_labels;
    std::vector<Label> volume_labels;

    auto GetVolumeLabels = Overload{
        [&volume_labels](UnitInput const& u) {
            for (auto const& v : u.volumes)
            {
                Label vl = v.label;
                if (vl.ext.empty())
                {
                    vl.ext = u.label.name;
                }
                volume_labels.push_back(std::move(vl));
            }
        },
        [&volume_labels](RectArrayInput const& r) {
            for (auto i : range(r.grid[to_int(Axis::x)].size() - 1))
            {
                for (auto j : range(r.grid[to_int(Axis::y)].size() - 1))
                {
                    for (auto k : range(r.grid[to_int(Axis::z)].size() - 1))
                    {
                        Label vl;
                        vl.name = std::string("{" + std::to_string(i) + ","
                                              + std::to_string(j) + ","
                                              + std::to_string(k) + "}");
                        vl.ext = r.label.name;
                        volume_labels.push_back(std::move(vl));
                    }
                }
            }
        }};

    auto GetSurfaceLabels = Overload{
        [&surface_labels](UnitInput const& u) {
            for (auto const& s : u.surfaces.labels)
            {
                Label surface_label = s;
                if (surface_label.ext.empty())
                {
                    surface_label.ext = u.label.name;
                }
                surface_labels.push_back(std::move(surface_label));
            }
        },
        [&surface_labels](RectArrayInput const& r) {
            for (auto ax : range(Axis::size_))
            {
                for (auto i : range(r.grid[to_int(ax)].size()))
                {
                    Label sl;
                    sl.name = std::string("{" + std::string(1, to_char(ax))
                                          + "," + std::to_string(i) + "}");
                    sl.ext = r.label.name;
                    surface_labels.push_back(std::move(sl));
                }
            }
        }};

    for (auto const& u : input.universes)
    {
        std::visit(GetSurfaceLabels, u);
        std::visit(GetVolumeLabels, u);
    }

    surf_labels_ = LabelIdMultiMap<SurfaceId>{std::move(surface_labels)};
    vol_labels_ = LabelIdMultiMap<VolumeId>{std::move(volume_labels)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
