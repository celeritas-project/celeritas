//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionMirror.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Device.hh"

#include "ParamsDataInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store and reference persistent collection groups on host and device.
 * \tparam P Params data collection group
 *
 * This should generally be an implementation detail of Params classes, which
 * are constructed on host and must have the same data both on host and device.
 * The template \c P must be a `FooData` class that:
 * - Is templated on ownership and memory space
 * - Has a templated assignment operator to copy from one space to another
 * - Has a boolean operator returning whether it's in a valid state.
 *
 * On assignment, it will copy the data to the device if the GPU is enabled.
 *
 * \par Example:
 * \code
 * class FooParams
 * {
 *   public:
 *     using CollectionDeviceRef = FooData<Ownership::const_reference,
 *                                         MemSpace::device>;
 *
 *     const CollectionDeviceRef& device_ref() const
 *     {
 *         return data_.device_ref();
 *     }
 *   private:
 *     CollectionMirror<FooData> data_;
 * };
 * \endcode
 *
 * \todo Rename ParamsDataStore
 */
template<template<Ownership, MemSpace> class P>
class CollectionMirror final : public ParamsDataInterface<P>
{
  public:
    //!@{
    //! \name Type aliases
    using HostValue = celeritas::HostVal<P>;
    using HostRef = celeritas::HostCRef<P>;
    using DeviceRef = celeritas::DeviceCRef<P>;
    //!@}

  public:
    //! Default constructor leaves the class in an "unassigned" state
    CollectionMirror() = default;

    // Construct from host data
    explicit inline CollectionMirror(HostValue&& host);

    //! Whether the data is assigned
    explicit operator bool() const { return static_cast<bool>(host_); }

    //! Access data on host
    HostRef const& host_ref() const final { return host_ref_; }

    //! Access data on device, if the device is enabled
    DeviceRef const& device_ref() const final { return device_ref_; }

    using ParamsDataInterface<P>::ref;

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
 * Construct by capturing host data.
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
}  // namespace celeritas
