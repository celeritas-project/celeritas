//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepParams.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/AuxInterface.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/io/Label.hh"

#include "../SDData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeoParamsInterface;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage params and state data for sensitive detectors.
 *
 */
class SDParams : public ParamsDataInterface<SDParamsData>,
                 public AuxParamsInterface
{
  public:
    using VecLabel = std::vector<Label>;

  public:
    SDParams(std::string output_label,
             VecLabel volume_labels,
             GeoParamsInterface const& geo);

    //!@{
    //! \name Data interface

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }
    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }
    //!@}

  private:
    std::string output_label_;
    VecLabel volume_labels_;
    std::vector<VolumeId> volume_ids_;
    CollectionMirror<SDParamsData> mirror_;
};
}  // namespace detail

}  // namespace celeritas
