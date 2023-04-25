//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionMirror.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for copying setup-time Collection groups to host and device.
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
    //! \name Type aliases
    using HostValue = celeritas::HostVal<P>;
    using HostRef = celeritas::HostCRef<P>;
    using DeviceRef = celeritas::DeviceCRef<P>;
    //!@}

  public:
    //! Default constructor leaves in an "unassigned" state
    CollectionMirror() = default;

    // Construct from host data
    explicit inline CollectionMirror(HostValue&& host);

    //! Whether the data is assigned
    explicit operator bool() const { return static_cast<bool>(host_); }

    // Get references to data after construction
    template<MemSpace M>
    inline P<Ownership::const_reference, M> const& ref() const;

    //!@{
    //! Deprecated alias to ref
    HostRef const& host() const { return this->host_ref(); }
    DeviceRef const& device() const { return this->device_ref(); }
    HostRef const& host_ref() const { return this->ref<MemSpace::host>(); }
    DeviceRef const& device_ref() const
    {
        return this->ref<MemSpace::device>();
    }
    //!@}

  private:
    HostValue host_;
    HostRef host_ref_;
    P<Ownership::value, MemSpace::device> device_;
    DeviceRef device_ref_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<template<Ownership, MemSpace> class P>
CollectionMirror<P>::CollectionMirror(HostValue&& host)
    : host_(std::move(host))
{
    CELER_EXPECT(host_);
    host_ref_ = host_;
    if (celeritas::device())
    {
        // Copy data to device and save reference
        device_ = host_;
        device_ref_ = device_;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get references to data after construction.
 *
 * Calling with "device" memspace will raise an exception if \c
 * celeritas::device is null (and device data wasn't set).
 */
template<template<Ownership, MemSpace> class P>
template<MemSpace M>
P<Ownership::const_reference, M> const& CollectionMirror<P>::ref() const
{
    if constexpr (M == MemSpace::host)
    {
        return host_ref_;
    }
    else if constexpr (M == MemSpace::device)
    {
        CELER_ENSURE(device_ref_);
        return device_ref_;
    }
    // "error #128-D: loop is not reachable"
#ifndef __NVCC__
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
