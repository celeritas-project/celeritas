//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/MagFieldMap.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"

#include "FieldMapData.hh"

namespace celeritas
{
namespace detail
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
    using ReadMap   = std::function<detail::FieldMapInput()>;
    using HostRef   = detail::FieldMapHostRef;
    using DeviceRef = detail::FieldMapDeviceRef;
    //@}

  public:
    // Construct with a magnetic field map
    MagFieldMap(ReadMap load_map);

    //! Access field map data on the host
    const HostRef& host_ref() const { return mirror_.host(); }

    //! Access field map data on the device
    const DeviceRef& device_ref() const { return mirror_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<detail::FieldMapData> mirror_;

  private:
    using HostValue = detail::FieldMapData<Ownership::value, MemSpace::host>;
    void build_data(const ReadMap&, HostValue*);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
