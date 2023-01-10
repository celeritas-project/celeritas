//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/MagFieldMap.hh
//---------------------------------------------------------------------------//
#pragma once

#include <functional>

#include "corecel/data/CollectionMirror.hh"

#include "FieldMapData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Set up a 2D MagFieldMap.
 */
class MagFieldMap
{
  public:
    //@{
    //! Type aliases
    using ReadMap = std::function<FieldMapInput()>;
    using HostRef = HostCRef<FieldMapData>;
    using DeviceRef = DeviceCRef<FieldMapData>;
    //@}

  public:
    // Construct with a magnetic field map
    explicit MagFieldMap(ReadMap load_map);

    //! Access field map data on the host
    HostRef const& host_ref() const { return mirror_.host(); }

    //! Access field map data on the device
    DeviceRef const& device_ref() const { return mirror_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<FieldMapData> mirror_;

  private:
    using HostValue = HostVal<FieldMapData>;
    void build_data(ReadMap const&, HostValue*);
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
