//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MapFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"

#include "FieldMapData.hh"
#include "RZFieldInput.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Set up a 2D MapFieldParams.
 */
class MapFieldParams
{
  public:
    //@{
    //! Type aliases
    using HostRef = HostCRef<FieldMapData>;
    using DeviceRef = DeviceCRef<FieldMapData>;
    using Input = RZFieldInput;
    //@}

  public:
    // Construct with a magnetic field map
    explicit MapFieldParams(Input const& inp);

    //! Access field map data on the host
    HostRef const& host_ref() const { return mirror_.host(); }

    //! Access field map data on the device
    DeviceRef const& device_ref() const { return mirror_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<FieldMapData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
