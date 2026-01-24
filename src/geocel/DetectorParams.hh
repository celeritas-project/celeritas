//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/DetectorParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
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
 *
 * \note See \c celeritas::VolumeIdBuilder for how to construct these easily.
 */
class DetectorParams final : public ParamsDataInterface<DetectorParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using VecVolId = std::vector<VolumeId>;
    //!@}

  public:
    //! Construct without detectors
    DetectorParams() = default;

    // Construct from detector input and volume params reference
    DetectorParams(VolumeParams const& volumes, inp::Detectors detectors);

    //! Whether any detectors are present
    bool empty() const { return !static_cast<bool>(mirror_); }

    //! Number of detectors
    DetectorId::size_type size() const { return detectors_.detectors.size(); }

    // Access detector ID based on volume ID
    inline DetectorId detector_id(VolumeId vol_id) const;

    // Access volume ID based on detector ID
    inline std::vector<VolumeId> const& volume_id(DetectorId det_id) const;

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
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
std::vector<VolumeId> const& DetectorParams::volume_id(DetectorId det_id) const
{
    CELER_EXPECT(det_id < this->size());
    return detectors_.detectors[det_id.get()].volumes;
}

DetectorId DetectorParams::detector_id(VolumeId vol_id) const
{
    return host_ref().detector_ids[vol_id];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
