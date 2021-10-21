//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CollectionMirror.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for copying setup-time Collection classes to host and device.
 *
 * This should generally be an implementation detail of Params classes, which
 * are constructed on host and must have the same data both on host and device.
 * The template `P` must be a `FooData` class that:
 * - Is templated on ownership and memory space
 * - Has a templated assignment operator to copy from one space to another
 * - Has a boolean operator returning whether it's in a valid state.
 *
 * On assignment, it will copy the data to the device if the GPU is enabled.
 *
 * Example:
 * \code
 * class FooParams
 * {
 *   public:
 *     using CollectionDeviceRef = FooData<Ownership::const_reference,
 *                                         MemSpace::device>;
 *
 *     const CollectionDeviceRef& device_ref() const
 *     {
 *         return data_.device();
 *     }
 *   private:
 *     CollectionMirror<FooData> data_;
 * };
 * \endcode
 */
template<template<Ownership, MemSpace> class P>
class CollectionMirror
{
  public:
    //!@{
    //! Type aliases
    using HostValue = P<Ownership::value, MemSpace::host>;
    using HostRef   = P<Ownership::const_reference, MemSpace::host>;
    using DeviceRef = P<Ownership::const_reference, MemSpace::device>;
    //!@}

  public:
    //! Default constructor leaves in an "unassigned" state
    CollectionMirror() = default;

    // Construct from host data
    explicit inline CollectionMirror(HostValue&& host);

    //! Whether the data is assigned
    explicit operator bool() const { return static_cast<bool>(host_); }

    //! Get host pointers after construction
    const HostRef& host() const
    {
        CELER_ENSURE(host_ref_);
        return host_ref_;
    }

    //! Get device pointers after construction
    const DeviceRef& device() const
    {
        CELER_ENSURE(device_ref_);
        return device_ref_;
    }

  private:
    HostValue                             host_;
    HostRef                               host_ref_;
    P<Ownership::value, MemSpace::device> device_;
    DeviceRef                             device_ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "CollectionMirror.i.hh"
