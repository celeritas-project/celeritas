//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZMapFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"

#include "RZMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RZMapFieldInput;

//---------------------------------------------------------------------------//
/*!
 * Set up a 2D RZMapFieldParams.
 *
 * The input values should be converted to the native unit system.
 */
class RZMapFieldParams final : public ParamsDataInterface<RZMapFieldParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using Input = RZMapFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit RZMapFieldParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access field map data on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    ParamsDataStore<RZMapFieldParamsData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
