//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.cc
//---------------------------------------------------------------------------//
#include "SDParams.hh"

#include <unordered_map>

#include "corecel/data/CollectionBuilder.hh"
// #include "geocel/GeantGeoParams.hh"
#include "geocel/VolumeCollectionBuilder.hh"
// #include "geocel/vg/VecgeomParams.hh"
// #include "orange/OrangeParams.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/geo/CoreGeoParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from list of volume labels.
 */
SDParams::SDParams(CoreGeoParams const& geo, inp::Detectors detectors)
    : detectors_{std::move(detectors)}
{
    if (!detectors_)
    {
        CELER_LOG(warning) << "Empty detectors list passed to SDParams";
    }

    // Map labels to volume IDs
    auto const num_impl_volumes = geo.impl_volumes().size();

    for (auto& detector : detectors_.detectors)
    {
        CELER_VALIDATE(std::all_of(detector.volumes.begin(),
                                   detector.volumes.end(),
                                   [num_impl_volumes](VolumeId id) {
                                       return id < num_impl_volumes;
                                   }),
                       << "invalid volume IDs given to SDParams");
    }

    std::unordered_map<VolumeId, DetectorId> detector_map;
    for (size_type det_num = 0; det_num < detectors_.detectors.size();
         ++det_num)
    {
        DetectorId det_id(det_num);
        auto& detector = detectors_.detectors[det_num];
        for (auto& volume : detector.volumes)
        {
            detector_map[volume] = det_id;
        }
    }

    mirror_ = CollectionMirror{[&] {
        HostVal<SDParamsData> host_data;
        host_data.detectors = build_volume_collection<DetectorId>(
            geo, VolumeMapFiller{detector_map});
        CELER_ENSURE(host_data);
        return host_data;
    }()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
