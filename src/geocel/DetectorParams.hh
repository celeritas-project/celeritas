//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/DetectorParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/cont/LabelIdMultiMap.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "DetectorData.hh"
#include "inp/Model.hh"

namespace celeritas
{
class VolumeParams;

//---------------------------------------------------------------------------//
/*!
 * Map Geant4 sensitive detectors to distinct detector IDs.
 */
class DetectorParams final : public ParamsDataInterface<DetectorParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SpanVol = Span<VolumeId const>;
    using DetectorMap = LabelIdMultiMap<DetectorId>;
    //!@}

  public:
    //! Construct without detectors or volumes
    DetectorParams() = default;

    // Construct from detector input and volume params reference
    DetectorParams(inp::Detectors detectors, VolumeParams const& volumes);

    //! Whether detector mapping is disabled (no volumes specified)
    bool empty() const { return !static_cast<bool>(mirror_); }

    //! Number of detectors
    DetectorId::size_type num_detectors() const
    {
        return detectors_.detectors.size();
    }

    //! Get detector metadata
    DetectorMap const& detector_labels() const { return det_labels_; }

    // Find the detector ID for a given volume, if any
    inline DetectorId detector_id(VolumeId vol_id) const;

    // Find all volumes assigned to a detector.
    inline SpanVol volume_ids(DetectorId det_id) const;

    //!@{
    //! \name Data interface

    //! Access sensitive detector properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }
    //! Access sensitive detector properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }
    //!@}

  private:
    CollectionMirror<DetectorParamsData> mirror_;
    inp::Detectors detectors_;
    DetectorMap det_labels_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Find the detector ID for a given volume, if any.
 */
DetectorId DetectorParams::detector_id(VolumeId vol_id) const
{
    return host_ref().detector_ids[vol_id];
}

//---------------------------------------------------------------------------//
/*!
 * Find all volumes assigned to a detector.
 */
auto DetectorParams::volume_ids(DetectorId det_id) const -> SpanVol
{
    CELER_EXPECT(det_id < this->num_detectors());
    return make_span(detectors_.detectors[det_id.get()].volumes);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
