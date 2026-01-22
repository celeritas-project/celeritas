//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "geocel/GeoParamsInterface.hh"
#include "geocel/inp/Model.hh"

#include "DetectorData.hh"

namespace celeritas
{
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

    // Construct from detector input and geometry reference
    DetectorParams(GeoParamsInterface const& geo, inp::Detectors detectors);

    //! Whether any detectors are present
    bool empty() const { return !static_cast<bool>(mirror_); }

    //! Number of detectors
    DetectorId::size_type size() const { return detectors_.detectors.size(); }

    //! Access detector ID based on implementation volume ID
    DetectorId volume_to_detector_id(ImplVolumeId iv_id)
    {
        return host_ref().detectors[iv_id];
    }

    //! Access volume ID based on detector ID
    std::vector<VolumeId> const& detector_to_volume_id(DetectorId det_id)
    {
        CELER_EXPECT(det_id < this->size());
        return detectors_.detectors[det_id.get()].volumes;
    }

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
}  // namespace celeritas
