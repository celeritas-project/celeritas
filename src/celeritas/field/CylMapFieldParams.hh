//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CylMapFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/data/ParamsDataStore.hh"

#include "CylMapFieldData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct CylMapFieldInput;

//---------------------------------------------------------------------------//
/*!
 * Set up a 3D nonuniform cylindrical field map.
 *
 * The field is interpolated on a cylindrical grid of \f$(r, \phi, z)\f$
 * coordinates, \em and the field itself is stored in cylindrical coordinates.
 */
class CylMapFieldParams final
    : public ParamsDataInterface<CylMapFieldParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using real_type = cylmap_real_type;
    using Input = CylMapFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit CylMapFieldParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access field map data on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    ParamsDataStore<CylMapFieldParamsData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
