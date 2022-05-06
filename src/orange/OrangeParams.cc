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
#include "corecel/cont/Array.hh"
#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/io/Logger.hh"
#include "construct/SurfaceInput.hh"
#include "construct/SurfaceInserter.hh"
#include "construct/VolumeInput.hh"
#include "construct/VolumeInserter.hh"
#include "univ/detail/LogicStack.hh"
#include "celeritas/Types.hh"

#include "Data.hh"
#include "Types.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/cont/Array.json.hh"

#    include "construct/SurfaceInputIO.json.hh"
#    include "construct/VolumeInputIO.json.hh"
#endif

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Load a geometry from the given JSON filename.
 */
OrangeParams::Input input_from_json(std::string filename)
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

    OrangeParams::Input input;

#if CELERITAS_USE_JSON
    std::ifstream infile(filename);
    CELER_VALIDATE(infile,
                   << "failed to open geometry at '" << filename << '\'');

    auto        full_inp  = nlohmann::json::parse(infile);
    const auto& universes = full_inp["universes"];

    CELER_VALIDATE(universes.size() == 1,
                   << "input geometry has " << universes.size()
                   << "universes; at present there must be a single global "
                      "universe");
    const auto& uni = universes[0];

    {
        // Insert surfaces
        SurfaceInserter insert(&input.surfaces);
        insert(uni["surfaces"].get<SurfaceInput>());
        uni["surface_names"].get_to(input.surface_labels);
    }

    {
        // Insert volumes
        auto           surface_ref = make_const_ref(input.surfaces);
        VolumeInserter insert(surface_ref, &input.volumes);
        for (const auto& vol_inp : uni["cells"])
        {
            insert(vol_inp.get<VolumeInput>());
        }
        uni["cell_names"].get_to(input.volume_labels);

        CELER_VALIDATE(static_cast<size_type>(insert.max_logic_depth())
                           < detail::LogicStack::max_stack_depth(),
                       << "input geometry has at least one volume with a "
                          "logic depth of"
                       << insert.max_logic_depth()
                       << " (surfaces are nested too deeply); but the logic "
                          "stack is limited to a depth of "
                       << detail::LogicStack::max_stack_depth());
    }

    // Add connectivity
    // TODO: calculate this on the fly using VolumeInserter
    CELER_VALIDATE(uni["surfaces"].contains("connectivity"),
                   << "input geometry is missing surface connectivity; "
                      "regenerate the JSON file with orange2celeritas");
    {
        // Volume ID storage
        auto volumes      = make_builder(&input.volumes.volumes);
        auto connectivity = make_builder(&input.volumes.connectivity);

        std::vector<VolumeId> temp_ids;
        for (const auto& surf_to_vol : uni["surfaces"]["connectivity"])
        {
            // Convert from values to IDs: a transform iterator would be a more
            // elegant way to do this
            temp_ids.resize(surf_to_vol.size());
            for (auto i : range(surf_to_vol.size()))
            {
                temp_ids[i] = VolumeId(surf_to_vol[i]);
            }
            Connectivity conn;
            conn.neighbors
                = volumes.insert_back(temp_ids.begin(), temp_ids.end());
            connectivity.push_back(conn);
        }
        CELER_ASSERT(input.volumes.connectivity.size()
                     == input.surfaces.size());
    }

    {
        // Save bbox
        const auto& bbox = uni["bbox"];
        input.bbox       = {bbox[0].get<Real3>(), bbox[1].get<Real3>()};
    }
#endif

    return input;
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
OrangeParams::OrangeParams(Input input)
{
    CELER_EXPECT(input.surfaces && input.volumes);
    CELER_EXPECT(input.surface_labels.size() == input.surfaces.size());
    CELER_EXPECT(input.volume_labels.size() == input.volumes.size());

    // Construct metadata
    surf_labels_ = std::move(input.surface_labels);
    vol_labels_  = std::move(input.volume_labels);
    for (auto sid : range(SurfaceId(surf_labels_.size())))
    {
        auto iter_inserted = surf_ids_.insert({surf_labels_[sid.get()], sid});
        CELER_VALIDATE(iter_inserted.second,
                       << "duplicate surface name '"
                       << iter_inserted.first->first << '\'');
    }
    for (auto vid : range(VolumeId(vol_labels_.size())))
    {
        auto iter_inserted = vol_ids_.insert({vol_labels_[vid.get()], vid});
        CELER_VALIDATE(iter_inserted.second,
                       << "duplicate volume name '"
                       << iter_inserted.first->first << '\'');
    }
    bbox_ = input.bbox;

    CELER_VALIDATE(input.volumes.connectivity.size() == input.surfaces.size(),
                   << "missing connectivity information");

    // Construct data
    OrangeParamsData<Ownership::value, MemSpace::host> host_data;
    host_data.surfaces = std::move(input.surfaces);
    host_data.volumes  = std::move(input.volumes);

    // Calculate max faces and intersections, reserving at least one to
    // improve error checking in state
    size_type max_faces         = 1;
    size_type max_intersections = 1;
    bool      simple_safety     = true;
    for (auto vol_id : range(VolumeId{host_data.volumes.size()}))
    {
        const VolumeRecord& def = host_data.volumes.defs[vol_id];
        max_faces = std::max<size_type>(max_faces, def.faces.size());
        max_intersections
            = std::max<size_type>(max_intersections, def.max_intersections);
        simple_safety &= (def.flags & VolumeRecord::simple_safety);
    }
    host_data.scalars.max_faces         = max_faces;
    host_data.scalars.max_intersections = max_intersections;

    if (!simple_safety)
    {
        CELER_LOG(warning) << "Geometry contains surfaces that are "
                              "incompatible with the current ORANGE simple "
                              "safety algorithm: multiple scattering may "
                              "result in arbitrarily small steps";
    }

    // Construct device values and device/host references
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<OrangeParamsData>{std::move(host_data)};

    CELER_ENSURE(data_);
    CELER_ENSURE(surf_labels_.size() == this->host_ref().surfaces.size());
    CELER_ENSURE(vol_labels_.size() == this->host_ref().volumes.size());
    CELER_ENSURE(surf_ids_.size() == surf_labels_.size());
    CELER_ENSURE(vol_ids_.size() == vol_labels_.size());
    CELER_ENSURE(bbox_);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a volume.
 */
const std::string& OrangeParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_[vol.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a label.
 *
 * If the label isn't in the geometry, a null ID will be returned.
 */
VolumeId OrangeParams::find_volume(const std::string& label) const
{
    auto iter = vol_ids_.find(label);
    if (iter == vol_ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Get the label of a surface.
 */
const std::string& OrangeParams::id_to_label(SurfaceId surf) const
{
    CELER_EXPECT(surf < surf_labels_.size());
    return surf_labels_[surf.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Locate the surface ID corresponding to a label.
 */
SurfaceId OrangeParams::find_surface(const std::string& label) const
{
    auto iter = surf_ids_.find(label);
    if (iter == surf_ids_.end())
        return {};
    return iter->second;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
