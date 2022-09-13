//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeParams.cc
//---------------------------------------------------------------------------//
#include "OrangeParams.hh"

#include <algorithm>
#include <fstream>
#include <initializer_list>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"

#include "Data.hh"
#include "Types.hh"
#include "construct/OrangeInput.hh"
#include "detail/SurfaceInserter.hh"
#include "detail/VolumeInserter.hh"
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
} // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a JSON file (if JSON is enabled).
 *
 * The JSON format is defined by the SCALE ORANGE exporter (not currently
 * distributed).
 */
OrangeParams::OrangeParams(const std::string& json_filename)
    : OrangeParams(input_from_json(json_filename))
{
}

//---------------------------------------------------------------------------//
/*!
 * Advanced usage: construct from explicit host data.
 *
 * Volume and surface labels must be unique for the time being.
 */
OrangeParams::OrangeParams(OrangeInput input)
{
    CELER_VALIDATE(input.units.size() == 1,
                   << "input geometry has " << input.units.size()
                   << "universes; at present there must be a single global "
                      "universe");
    UnitInput& u = input.units.front();
    CELER_VALIDATE(u,
                   << "unit '" << u.label << "' is not properly constructed");
    {
        // Capture metadata
        surf_labels_ = LabelIdMultiMap<SurfaceId>{std::move(u.surfaces.labels)};
        std::vector<Label> volume_labels;
        volume_labels.resize(u.volumes.size());
        for (auto i : range(u.volumes.size()))
        {
            volume_labels[i] = std::move(u.volumes[i].label);
        }
        vol_labels_ = LabelIdMultiMap<VolumeId>{std::move(volume_labels)};
        bbox_       = u.bbox;
    }

    // Construct data
    HostVal<OrangeParamsData> host_data;
    {
        // Insert surfaces
        detail::SurfaceInserter insert(&host_data);
        insert(u.surfaces);
    }

    {
        // Insert volumes
        detail::VolumeInserter insert(&host_data);
        std::for_each(u.volumes.begin(), u.volumes.end(), insert);

        CELER_VALIDATE(static_cast<size_type>(insert.max_logic_depth())
                           < detail::LogicStack::max_stack_depth(),
                       << "input geometry has at least one volume with a "
                          "logic depth of"
                       << insert.max_logic_depth()
                       << " (surfaces are nested too deeply); but the logic "
                          "stack is limited to a depth of "
                       << detail::LogicStack::max_stack_depth());
    }

    {
        // Add connectivity
        CELER_VALIDATE(u.connectivity.size() == u.surfaces.size(),
                       << "missing connectivity information");

        // Volume ID storage
        auto volumes      = make_builder(&host_data.volumes.volumes);
        auto connectivity = make_builder(&host_data.volumes.connectivity);

        for (const auto& ids : u.connectivity)
        {
            Connectivity conn;
            conn.neighbors = volumes.insert_back(ids.begin(), ids.end());
            connectivity.push_back(conn);
        }
        CELER_ASSERT(host_data.volumes.connectivity.size()
                     == host_data.surfaces.size());
    }

    // Calculate max faces and intersections
    // (Use 1 instead of 0 in case of trivial geometries to improve assertions)
    size_type max_faces         = 1;
    size_type max_intersections = 1;
    bool      simple_safety     = true;
    for (auto vol_id : range(VolumeId{host_data.volumes.size()}))
    {
        const VolumeRecord& def = host_data.volumes.defs[vol_id];
        max_faces = std::max<size_type>(max_faces, def.faces.size());
        max_intersections
            = std::max<size_type>(max_intersections, def.max_intersections);
        simple_safety = simple_safety && supports_simple_safety(def.flags);
    }
    host_data.scalars.max_faces         = max_faces;
    host_data.scalars.max_intersections = max_intersections;
    supports_safety_ = simple_safety;

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};

    CELER_ENSURE(data_);
    CELER_ENSURE(surf_labels_.size() == this->host_ref().surfaces.size());
    CELER_ENSURE(vol_labels_.size() == this->host_ref().volumes.size());
    CELER_ENSURE(bbox_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a volume.
 */
const Label& OrangeParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_.get(vol);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a label.
 *
 * If the label isn't in the geometry, a null ID will be returned.
 */
VolumeId OrangeParams::find_volume(const std::string& name) const
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
 * Get zero or more volume IDs corresponding to a name.
 *
 * This is useful for volumes that are repeated in the geometry with different
 * uniquifying 'extensions'.
 */
auto OrangeParams::find_volumes(const std::string& name) const
    -> SpanConstVolumeId
{
    return vol_labels_.find_all(name);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a surface.
 */
const Label& OrangeParams::id_to_label(SurfaceId surf) const
{
    CELER_EXPECT(surf < surf_labels_.size());
    return surf_labels_.get(surf);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the surface ID corresponding to a label name.
 */
SurfaceId OrangeParams::find_surface(const std::string& name) const
{
    auto result = surf_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "surface '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
} // namespace celeritas
