//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/DetectorParams.cc
//---------------------------------------------------------------------------//
#include "DetectorParams.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "geocel/Types.hh"

#include "VolumeParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from geometry and detector input.
 */
DetectorParams::DetectorParams(inp::Detectors idets,
                               VolumeParams const& volumes)
    : detectors_{std::move(idets)}
{
    CELER_EXPECT(!volumes.empty());

    // Build label mapping
    std::vector<Label> labels;
    labels.reserve(detectors_.detectors.size());
    for (auto const& detector : detectors_.detectors)
    {
        labels.push_back(detector.label);
    }
    det_labels_ = DetectorMap{std::move(labels)};

    // Map volumes to detectors and validate
    std::vector<DetectorId> dets(volumes.num_volumes(), DetectorId{});
    for (DetectorId det_id : range(DetectorId{this->num_detectors()}))
    {
        auto const& detector = detectors_.detectors[det_id.get()];
        for (VolumeId vol_id : detector.volumes)
        {
            CELER_VALIDATE(vol_id < dets.size(),
                           << "out-of-range volume ID "
                           << vol_id.unchecked_get() << " in detector "
                           << detector.label);
            CELER_VALIDATE(!dets[vol_id.get()],
                           << "volume " << vol_id.unchecked_get()
                           << " assigned to multiple detectors");
            dets[vol_id.get()] = det_id;
        }
    }

    mirror_ = CollectionMirror{[&dets] {
        HostVal<DetectorParamsData> host_data;

        CollectionBuilder{&host_data.detector_ids}.insert_back(dets.begin(),
                                                               dets.end());
        CELER_ENSURE(host_data);
        return host_data;
    }()};

    CELER_ENSURE(det_labels_.size() == detectors_.detectors.size());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
