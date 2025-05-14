//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/SDParams.cc
//---------------------------------------------------------------------------//
#include "SDParams.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Join.hh"
#include "geocel/GeoVolumeFinder.hh"

namespace celeritas
{

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from list of volume labels.
 */
SDParams::SDParams(VecLabel volume_labels, GeoParamsInterface const& geo)
{
    CELER_EXPECT(!volume_labels.empty());

    // Map labels to volume IDs
    volume_ids_.resize(volume_labels.size());

    std::vector<std::reference_wrapper<Label const>> missing;
    GeoVolumeFinder find_volume(geo);
    for (auto i : range(volume_labels.size()))
    {
        volume_ids_[i] = find_volume(volume_labels[i]);
        if (!volume_ids_[i])
        {
            missing.emplace_back(volume_labels[i]);
        }
    }

    CELER_VALIDATE(missing.empty(),
                   << "failed to find " << cmake::core_geo
                   << " volume(s) for labels '"
                   << join(missing.begin(), missing.end(), "', '"));
    CELER_ENSURE(volume_ids_.size() == volume_labels.size());

    std::map<VolumeId, DetectorId> detector_map;
    for (auto didx : range<DetectorId::size_type>(volume_ids_.size()))
    {
        detector_map[volume_ids_[didx]] = DetectorId{didx};
    }

    mirror_ = CollectionMirror{[&] {
        HostVal<SDParamsData> host_data;
        if (!detector_map.empty())
        {
            std::vector<DetectorId> temp_det(geo.volumes().size(),
                                             DetectorId{});
            for (auto const& det_pair : detector_map)
            {
                CELER_ASSERT(det_pair.first < temp_det.size());
                temp_det[det_pair.first.unchecked_get()] = det_pair.second;
            }
            CollectionBuilder{&host_data.detector}.insert_back(
                temp_det.begin(), temp_det.end());
        }

        return host_data;
    }()};
}

DetectorId SDParams::volume_to_detector_id(VolumeId vol_id)
{
    return host_ref().detector[vol_id];
}

VolumeId SDParams::detector_to_volume_id(DetectorId det_id)
{
    return volume_ids_[det_id.get()];
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
