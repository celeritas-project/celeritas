//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CylFieldMapParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "CylFieldMapData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct CylFieldMapInput;

//---------------------------------------------------------------------------//
/*!
 * Set up a 3D CylFieldMapParams.
 *
 * The input values are in the native unit system.
 */
class CylFieldMapParams final
    : public ParamsDataInterface<CylFieldMapParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using Input = CylFieldMapInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit CylFieldMapParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access field map data on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<CylFieldMapParamsData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
