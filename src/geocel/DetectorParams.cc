//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/DetectorParams.cc
//---------------------------------------------------------------------------//
#include "DetectorParams.hh"

#include <unordered_map>

#include "GeoParamsInterface.hh"
#include "VolumeCollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from geometry and detector input.
 */
DetectorParams::DetectorParams(GeoParamsInterface const& geo,
                               inp::Detectors detectors)
    : detectors_{std::move(detectors)}
{
    // Map labels to volume IDs
    auto const num_impl_volumes = geo.impl_volumes().size();

    for (auto& detector : detectors_.detectors)
    {
        CELER_VALIDATE(std::all_of(detector.volumes.begin(),
                                   detector.volumes.end(),
                                   [num_impl_volumes](VolumeId id) {
                                       return id < num_impl_volumes;
                                   }),
                       << "invalid volume IDs given to DetectorParams");
    }

    std::unordered_map<VolumeId, DetectorId> detector_map;
    for (auto det_id : range(id_cast<DetectorId>(detectors_.detectors.size())))
    {
        auto& detector = detectors_.detectors[det_id.get()];
        for (auto& volume : detector.volumes)
        {
            detector_map[volume] = det_id;
        }
    }

    mirror_ = CollectionMirror{[&] {
        HostVal<DetectorParamsData> host_data;
        host_data.detectors = build_volume_collection<DetectorId>(
            geo, VolumeMapFiller{detector_map});
        CELER_ENSURE(host_data);
        return host_data;
    }()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
