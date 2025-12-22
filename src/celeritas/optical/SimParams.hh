//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SimParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/inp/Tracking.hh"

#include "SimData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Manage persistent simulation data.
 */
class SimParams final : public ParamsDataInterface<SimParamsData>
{
  public:
    // Construct with simulation input data
    explicit SimParams(inp::OpticalTrackingLimits const&);

    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    //! Maximum step iterations before aborting
    size_type max_step_iters() const
    {
        return this->host_ref().max_step_iters;
    }

  private:
    // Host/device storage and reference
    CollectionMirror<SimParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
