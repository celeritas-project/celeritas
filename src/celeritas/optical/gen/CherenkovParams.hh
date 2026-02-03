//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/CherenkovParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"

#include "CherenkovData.hh"

namespace celeritas
{
namespace optical
{
class MaterialParams;
}

//---------------------------------------------------------------------------//
/*!
 * Build and manage Cherenkov data.
 */
class CherenkovParams final : public ParamsDataInterface<CherenkovData>
{
  public:
    // Construct with optical property data
    explicit CherenkovParams(optical::MaterialParams const& material);

    //! Access physics material on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access physics material on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    ParamsDataStore<CherenkovData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
