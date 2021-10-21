//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MagFieldMap.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/CollectionMirror.hh"
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
    using ReadMap   = std::function<detail::FieldMapData()>;
    using HostRef   = detail::FieldMapHostRef;
    using DeviceRef = detail::FieldMapDeviceRef;
    //@}

  public:
    // Construct with a magnetic field map
    MagFieldMap(ReadMap load_map);

    //! Access field map data on the host
    const HostRef& host_group() const { return group_.host(); }

    //! Access field map data on the device
    const DeviceRef& device_group() const { return group_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<detail::FieldMapGroup> group_;

  private:
    using HostValue = detail::FieldMapGroup<Ownership::value, MemSpace::host>;
    void build_data(ReadMap load_map, HostValue* group);
};

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
