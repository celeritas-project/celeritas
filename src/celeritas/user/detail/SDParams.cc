//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepParams.cc
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
SDParams::SDParams(std::string output_label,
                   VecLabel labels,
                   GeoParamsInterface const& geo)
    : output_label_{std::move(output_label)}, volume_labels_{std::move(labels)}
{
    CELER_EXPECT(!output_label_.empty());
    CELER_EXPECT(!volume_labels_.empty());

    enum class HasDetectors
    {
        unknown = -1,
        none,
        all
    };

    HasDetectors has_det = HasDetectors::unknown;

    // Map labels to volume IDs
    volume_ids_.resize(volume_labels_.size());

    std::vector<std::reference_wrapper<Label const>> missing;
    GeoVolumeFinder find_volume(geo);
    for (auto i : range(volume_labels_.size()))
    {
        volume_ids_[i] = find_volume(volume_labels_[i]);
        if (!volume_ids_[i])
        {
            missing.emplace_back(volume_labels_[i]);
        }
    }

    CELER_VALIDATE(missing.empty(),
                   << "failed to find " << cmake::core_geo
                   << " volume(s) for labels '"
                   << join(missing.begin(), missing.end(), "', '"));
    CELER_ENSURE(volume_ids_.size() == volume_labels_.size());

    std::map<VolumeId, DetectorId> detector_map;
    for (auto didx : range<DetectorId::size_type>(volume_ids_.size()))
    {
        detector_map[volume_ids_[didx]] = DetectorId{didx};
    }

    auto this_has_detectors = detector_map.empty() ? HasDetectors::none
                                                   : HasDetectors::all;
    if (has_det == HasDetectors::unknown)
    {
        has_det = this_has_detectors;
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

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas