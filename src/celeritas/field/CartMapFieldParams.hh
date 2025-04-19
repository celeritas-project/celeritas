//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartMapFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "CartMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct CartMapFieldInput;

//---------------------------------------------------------------------------//
/*!
 * Set up a 3D CartMapFieldParams.
 *
 * The input values are in the native unit system.
 */
class CartMapFieldParams final
    : public ParamsDataInterface<CartMapFieldParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using real_type = cartmap_real_type;
    using Input = CartMapFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit CartMapFieldParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const final { return host_ref_; }

    //! Access field map data on the device
    DeviceRef const& device_ref() const final { return device_ref_; }

  private:
    // Host/device storage and reference
    HostVal<CartMapFieldParamsData> host_;
    HostCRef<CartMapFieldParamsData> host_ref_;
    CartMapFieldParamsData<Ownership::value, MemSpace::device> device_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
