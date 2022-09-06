//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterialParams.cc
//---------------------------------------------------------------------------//
#include "GeoMaterialParams.hh"

#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "orange/OrangeParams.hh"
#include "orange/Types.hh"

#include "GeoMaterialData.hh"
#include "GeoParams.hh"

namespace celeritas
{
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

    if (!input.volume_labels.empty())
    {
        // Remap materials to volume IDs using given volume names:
        // build a map of volume name -> matid
        std::unordered_map<Label, MaterialId> lab_to_id;
        std::set<Label>                       duplicates;
        for (auto idx : range(input.volume_to_mat.size()))
        {
            auto iter_inserted
                = lab_to_id.insert({std::move(input.volume_labels[idx]),
                                    input.volume_to_mat[idx]});
            if (!iter_inserted.second)
            {
                duplicates.insert(iter_inserted.first->first);
            }
        }
        CELER_VALIDATE(duplicates.empty(),
                       << "geo/material coupling specified duplicate "
                          "volume names: \""
                       << join(duplicates.begin(), duplicates.end(), "\", \"")
                       << '"');

        // Set material ids based on volume names
        std::vector<Label> missing_volumes;
        const GeoParams&   geo = *input.geometry;
        input.volume_to_mat.assign(geo.num_volumes(), MaterialId{});
        for (auto volume_id : range(VolumeId{geo.num_volumes()}))
        {
            auto iter = lab_to_id.find(geo.id_to_label(volume_id));
            if (iter == lab_to_id.end())
            {
                const Label& label = geo.id_to_label(volume_id);
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
    auto      materials = make_builder(&host_data.materials);
    materials.insert_back(input.volume_to_mat.begin(),
                          input.volume_to_mat.end());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<GeoMaterialParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
