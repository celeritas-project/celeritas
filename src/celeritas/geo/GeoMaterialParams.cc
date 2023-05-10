//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterialParams.cc
//---------------------------------------------------------------------------//
#include "GeoMaterialParams.hh"

#include <algorithm>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/detail/Joined.hh"
#include "corecel/sys/ScopedMem.hh"
#include "orange/Types.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/io/ImportData.hh"

#include "GeoMaterialData.hh"  // IWYU pragma: associated

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<GeoMaterialParams>
GeoMaterialParams::from_import(ImportData const& data,
                               SPConstGeo geo_params,
                               SPConstMaterial material_params)
{
    GeoMaterialParams::Input input;
    input.geometry = std::move(geo_params);
    input.materials = std::move(material_params);

    input.volume_to_mat.resize(data.volumes.size());
    for (auto volume_idx :
         range<VolumeId::size_type>(input.volume_to_mat.size()))
    {
        if (!data.volumes[volume_idx])
            continue;

        input.volume_to_mat[volume_idx]
            = MaterialId(data.volumes[volume_idx].material_id);
    }

    // Assume that since Geant4 is using internal geometry and
    // we're using ORANGE or VecGeom that volume IDs will not be
    // the same. We'll just remap them based on their labels (which may include
    // Geant4's uniquifying pointer addresses).
    input.volume_labels.resize(data.volumes.size());
    for (auto volume_idx : range(data.volumes.size()))
    {
        if (!data.volumes[volume_idx])
            continue;

        CELER_EXPECT(!data.volumes[volume_idx].name.empty());
        input.volume_labels[volume_idx]
            = Label::from_geant(data.volumes[volume_idx].name);
    }

    return std::make_shared<GeoMaterialParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct from geometry and material params.
 *
 * Missing material IDs may be allowed if they correspond to unreachable volume
 * IDs.
 */
GeoMaterialParams::GeoMaterialParams(Input input)
{
    CELER_EXPECT(input.geometry);
    CELER_EXPECT(input.materials);
    CELER_EXPECT(
        (input.volume_labels.empty()
         && input.volume_to_mat.size() == input.geometry->num_volumes())
        || input.volume_to_mat.size() == input.volume_labels.size());
    CELER_EXPECT(std::all_of(input.volume_to_mat.begin(),
                             input.volume_to_mat.end(),
                             [&input](MaterialId m) {
                                 return !m
                                        || m < input.materials->num_materials();
                             }));

    ScopedMem record_mem("GeoMaterialParams.construct");

    if (!input.volume_labels.empty())
    {
        // Remap materials to volume IDs using given volume names:
        // build a map of volume name -> matid
        std::unordered_map<Label, MaterialId> lab_to_id;
        std::set<Label> duplicates;
        for (auto idx : range(input.volume_to_mat.size()))
        {
            if (!input.volume_to_mat[idx])
            {
                // Skip volumes without matids
                continue;
            }

            auto [prev, inserted]
                = lab_to_id.insert({std::move(input.volume_labels[idx]),
                                    input.volume_to_mat[idx]});
            if (!inserted)
            {
                duplicates.insert(prev->first);
            }
        }
        CELER_VALIDATE(duplicates.empty(),
                       << "geo/material coupling specified duplicate "
                          "volume names: \""
                       << join(duplicates.begin(), duplicates.end(), "\", \"")
                       << '"');

        // Set material ids based on volume names
        std::vector<Label> missing_volumes;
        GeoParams const& geo = *input.geometry;
        input.volume_to_mat.assign(geo.num_volumes(), MaterialId{});
        for (auto volume_id : range(VolumeId{geo.num_volumes()}))
        {
            auto iter = lab_to_id.find(geo.id_to_label(volume_id));
            if (iter == lab_to_id.end())
            {
                // Exact label (name + ext) not found; fall back to matching
                // just the name
                iter = lab_to_id.find(Label{geo.id_to_label(volume_id).name});
            }

            if (iter == lab_to_id.end())
            {
                Label const& label = geo.id_to_label(volume_id);
                if (!label.name.empty()
                    && !(label.name.front() == '[' && label.name.back() == ']'))
                {
                    // Skip "[unused]" that we set for vecgeom empty labels,
                    // "[EXTERIOR]" from ORANGE
                    missing_volumes.push_back(label);
                }
            }
            else
            {
                input.volume_to_mat[volume_id.unchecked_get()] = iter->second;
            }
        }
        if (!missing_volumes.empty())
        {
            CELER_LOG(warning)
                << "Some geometry volumes do not have known material IDs: "
                << join(missing_volumes.begin(), missing_volumes.end(), ", ");
        }
    }

    HostValue host_data;
    auto materials = make_builder(&host_data.materials);
    materials.insert_back(input.volume_to_mat.begin(),
                          input.volume_to_mat.end());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<GeoMaterialParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
