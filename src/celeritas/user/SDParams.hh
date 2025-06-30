//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Label.hh"

#include "SDData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeoParamsInterface;

//---------------------------------------------------------------------------//
/*!
 * Manage params and state data for sensitive detectors.
 */
class SDParams final : public ParamsDataInterface<SDParamsData>
{
  public:
    using VecLabel = std::vector<Label>;

  public:
    //! Default Constructor
    SDParams() {}

    //! Construct from volume labels
    SDParams(VecLabel const& volume_labels, GeoParamsInterface const& geo);

    //! Whether any detectors are present
    bool empty() const { return !static_cast<bool>(mirror_); }

    //! Number of detectors
    DetectorId::size_type size() const { return volume_ids_.size(); }

    //! Access detector ID based on volume ID
    DetectorId volume_to_detector_id(VolumeId vol_id)
    {
        return host_ref().detector[vol_id];
    }

    //! Access volume ID based on detector ID
    VolumeId detector_to_volume_id(DetectorId det_id)
    {
        CELER_EXPECT(det_id < this->size());
        return volume_ids_[det_id.get()];
    }

    //!@{
    //! \name Data interface

    //! Access sensitive detector properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }
    //! Access sensitive detector properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }
    //!@}

  private:
    std::vector<VolumeId> volume_ids_;
    CollectionMirror<SDParamsData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
