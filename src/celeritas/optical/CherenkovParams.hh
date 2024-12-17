//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/CherenkovParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "CherenkovData.hh"

namespace celeritas
{
namespace optical
{
class MaterialParams;

//---------------------------------------------------------------------------//
/*!
 * Build and manage Cherenkov data.
 */
class CherenkovParams final : public ParamsDataInterface<CherenkovData>
{
  public:
    // Construct with optical property data
    explicit CherenkovParams(MaterialParams const& material);

    //! Access physics material on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access physics material on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    CollectionMirror<CherenkovData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
