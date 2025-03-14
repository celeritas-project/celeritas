//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/RZPhiMapFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "RZPhiMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct RZPhiMapFieldInput;

//---------------------------------------------------------------------------//
/*!
 * Set up a 3D RZPhiMapFieldParams.
 *
 * The input values should be converted to the native unit system.
 */
class RZPhiMapFieldParams final
    : public ParamsDataInterface<RZPhiMapFieldParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using Input = RZPhiMapFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit RZPhiMapFieldParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access field map data on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<RZPhiMapFieldParamsData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
