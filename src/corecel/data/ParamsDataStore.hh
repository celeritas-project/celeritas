//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ParamsDataStore.hh
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
 *     ParamsDataStore<FooData> data_;
 * };
 * \endcode
 */
template<template<Ownership, MemSpace> class P>
class ParamsDataStore final : public ParamsDataInterface<P>
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
    ParamsDataStore() = default;

    // Construct from host data
    explicit inline ParamsDataStore(HostValue&& host);

    //! Whether the data is assigned
    explicit operator bool() const { return static_cast<bool>(host_); }

    //! Access data on host
    CELER_FORCEINLINE HostRef const& host_ref() const final
    {
        return host_ref_;
    }

    //! Access data on device, if the device is enabled
    CELER_FORCEINLINE DeviceRef const& device_ref() const final
    {
        return device_ref_;
    }

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
ParamsDataStore<P>::ParamsDataStore(HostValue&& host) : host_(std::move(host))
{
    if (CELER_UNLIKELY(!host_))
    {
        CELER_DEBUG_FAIL("incomplete host data or bad copy", precondition);
    }

    host_ref_ = host_;

    if (celeritas::device())
    {
        if constexpr (!CELER_USE_DEVICE)
        {
            // Mark unreachable for optimization and coverage
            CELER_ASSERT_UNREACHABLE();
        }

        // Copy data to device and save reference
        device_ = host_;
        device_ref_ = device_;
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
