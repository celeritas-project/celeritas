//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "SDData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeoParamsInterface;

//---------------------------------------------------------------------------//
/*!
 * Map Geant4 sensitive detectors to distinct detector IDs.
 *
 * \note See \c celeritas::VolumeIdBuilder for how to construct these easily.
 */
class SDParams final : public ParamsDataInterface<SDParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using VecVolId = std::vector<VolumeId>;
    //!@}

  public:
    //! Default constructor: no detectors
    SDParams() = default;

    //! Construct from canonical volume IDs
    SDParams(GeoParamsInterface const& geo, VecVolId&& volume_ids);

    //! Whether any detectors are present
    bool empty() const { return !static_cast<bool>(mirror_); }

    //! Number of detectors
    DetectorId::size_type size() const { return volume_ids_.size(); }

    //! Access detector ID based on implementation volume ID
    DetectorId volume_to_detector_id(ImplVolumeId iv_id)
    {
        return host_ref().detector[iv_id];
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
    VecVolId volume_ids_;
    CollectionMirror<SDParamsData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
