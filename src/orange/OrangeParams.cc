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

#include "OrangeData.hh" // IWYU pragma: associated
#include "OrangeTypes.hh"
#include "construct/OrangeInput.hh"
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
 * Construct in-memory from a Geant4 geometry (not implemented).
 *
 * Perhaps someday we'll implement in-memory translation...
 */
OrangeParams::OrangeParams(const G4VPhysicalVolume*)
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
    CELER_VALIDATE(!input.units.empty(), << "input geometry has no universes");

    HostVal<OrangeParamsData> host_data;

    // Calculate offsets for UnitIndexerData
    auto ui_surf = make_builder(&host_data.unit_indexer_data.surfaces);
    auto ui_vol  = make_builder(&host_data.unit_indexer_data.volumes);
    ui_surf.push_back(0);
    ui_vol.push_back(0);
    for (const UnitInput& u : input.units)
    {
        using AllVals = AllItems<size_type, MemSpace::native>;
        auto surface_offset
            = host_data.unit_indexer_data.surfaces[AllVals{}].back();
        auto volume_offset
            = host_data.unit_indexer_data.volumes[AllVals{}].back();
        ui_surf.push_back(surface_offset + u.surfaces.size());
        ui_vol.push_back(volume_offset + u.volumes.size());
    }

    // Insert all units
    detail::UnitInserter insert_unit(&host_data);
    auto universe_type  = make_builder(&host_data.universe_type);
    auto universe_index = make_builder(&host_data.universe_index);

    for (const UnitInput& u : input.units)
    {
        CELER_VALIDATE(
            u, << "unit '" << u.label << "' is not properly constructed");
        SimpleUnitId uid = insert_unit(u);
        universe_type.push_back(UniverseType::simple);
        universe_index.push_back(uid.get());
    }
    CELER_VALIDATE(host_data.scalars.max_logic_depth
                       < detail::LogicStack::max_stack_depth(),
                   << "input geometry has at least one volume with a "
                      "logic depth of"
                   << host_data.scalars.max_logic_depth
                   << " (surfaces are nested too deeply); but the logic "
                      "stack is limited to a depth of "
                   << detail::LogicStack::max_stack_depth());

    std::vector<Label> surface_labels;
    std::vector<Label> volume_labels;

    for (const UnitInput& u : input.units)
    {
        // Capture metadata

        for (const auto& s : u.surfaces.labels)
        {
            Label surface_label = s;
            if (surface_label.ext.empty())
            {
                surface_label.ext = u.label.name;
            }
            surface_labels.push_back(std::move(surface_label));
        }

        for (const auto& v : u.volumes)
        {
            Label volume_label = v.label;
            if (volume_label.ext.empty())
            {
                volume_label.ext = u.label.name;
            }
            volume_labels.push_back(std::move(volume_label));
        }

        bbox_ = u.bbox;
    }

    surf_labels_ = LabelIdMultiMap<SurfaceId>{std::move(surface_labels)};
    vol_labels_  = LabelIdMultiMap<VolumeId>{std::move(volume_labels)};

    supports_safety_ = host_data.simple_unit[SimpleUnitId{0}].simple_safety;
    bbox_            = input.units.front().bbox;

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
const Label& OrangeParams::id_to_label(VolumeId vol) const
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
 * Locate the volume ID corresponding to a label.
 *
 * If the label isn't in the geometry, a null ID will be returned.
 */
VolumeId OrangeParams::find_volume(const Label& label) const
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
