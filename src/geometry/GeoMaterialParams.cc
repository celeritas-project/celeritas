//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GeoMaterialParams.cc
//---------------------------------------------------------------------------//
#include "GeoMaterialParams.hh"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

#include "base/Assert.hh"
#include "base/CollectionBuilder.hh"
#include "base/Join.hh"
#include "base/OpaqueId.hh"
#include "base/Range.hh"
#include "comm/Logger.hh"
#include "geometry/GeoMaterialData.hh"
#include "geometry/Types.hh"
#include "orange/OrangeParams.hh"

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
        (input.volume_names.empty()
         && input.volume_to_mat.size() == input.geometry->num_volumes())
        || input.volume_to_mat.size() == input.volume_names.size());
    CELER_EXPECT(std::all_of(input.volume_to_mat.begin(),
                             input.volume_to_mat.end(),
                             [&input](MaterialId m) {
                                 return !m
                                        || m < input.materials->num_materials();
                             }));

    if (!input.volume_names.empty())
    {
        // Remap materials to volume IDs using given volume names:
        // build a map of volume name -> matid
        std::unordered_map<std::string, MaterialId> name_to_id;
        for (auto idx : range(input.volume_to_mat.size()))
        {
            auto iter_inserted = name_to_id.insert(
                {std::move(input.volume_names[idx]), input.volume_to_mat[idx]});
            CELER_VALIDATE(iter_inserted.second,
                           << "geo/material coupling specified duplicate "
                              "volume name '"
                           << iter_inserted.first->first << "'");
        }

        // Set material ids based on volume names
        std::vector<std::string> missing_volumes;
        const GeoParams&         geo = *input.geometry;
        input.volume_to_mat.assign(geo.num_volumes(), MaterialId{});
        for (auto volume_id : range(VolumeId{geo.num_volumes()}))
        {
            auto iter = name_to_id.find(geo.id_to_label(volume_id));
            if (iter == name_to_id.end())
            {
                missing_volumes.push_back(geo.id_to_label(volume_id));
                continue;
            }
            input.volume_to_mat[volume_id.unchecked_get()] = iter->second;
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
