//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.cc
//---------------------------------------------------------------------------//
#include "SDParams.hh"

#include <unordered_map>

#include "corecel/data/CollectionBuilder.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/VolumeCollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from list of volume labels.
 */
SDParams::SDParams(GeoParamsInterface const& geo, VecVolId&& volume_ids)
    : volume_ids_{std::move(volume_ids)}
{
    CELER_EXPECT(!volume_ids_.empty());

    // Map labels to volume IDs
    auto const num_impl_volumes = geo.impl_volumes().size();

    CELER_VALIDATE(std::all_of(volume_ids_.begin(),
                               volume_ids_.end(),
                               [num_impl_volumes](VolumeId id) {
                                   return id < num_impl_volumes;
                               }),
                   << "invalid volume IDs given to SDParams");

    std::unordered_map<VolumeId, DetectorId> detector_map;
    for (auto didx : range<DetectorId::size_type>(volume_ids_.size()))
    {
        detector_map[volume_ids_[didx]] = DetectorId{didx};
    }

    mirror_ = CollectionMirror{[&] {
        HostVal<SDParamsData> host_data;
        host_data.detector = build_volume_collection<DetectorId>(
            geo, VolumeMapFiller{detector_map});
        CELER_ENSURE(host_data);
        return host_data;
    }()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
