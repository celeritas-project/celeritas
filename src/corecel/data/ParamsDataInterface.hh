//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ParamsDataInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interface class for accessing parameter data.
 */
template<template<Ownership, MemSpace> class P>
class ParamsDataInterface
{
  public:
    //!@{
    //! \name Type aliases
    using HostRef = HostCRef<P>;
    using DeviceRef = DeviceCRef<P>;
    //!@}

    //! Reference CPU geometry data
    virtual HostRef const& host_ref() const = 0;

    //! Reference managed GPU geometry data
    virtual DeviceRef const& device_ref() const = 0;

    // Dispatch a "ref" call to host or device data
    template<MemSpace M>
    inline P<Ownership::const_reference, M> const& ref() const;

  protected:
    // Protected destructor prevents deletion of pointer-to-interface
    ~ParamsDataInterface() = default;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Dispatch a "ref" call to host or device data.
 */
template<template<Ownership, MemSpace> class P>
template<MemSpace M>
P<Ownership::const_reference, M> const& ParamsDataInterface<P>::ref() const
{
    if constexpr (M == MemSpace::host)
    {
        auto const& result = this->host_ref();
        CELER_ENSURE(result);
        return result;
    }
    else if constexpr (M == MemSpace::device)
    {
        auto const& result = this->device_ref();
        CELER_ENSURE(result);
        return result;
    }
    // "error #128-D: loop is not reachable"
#ifndef __NVCC__
    CELER_ASSERT_UNREACHABLE();
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
