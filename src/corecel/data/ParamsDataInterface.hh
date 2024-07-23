//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ParamsDataInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "Ref.hh"

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

    // Prohibit copy/move beween interface classes
    ParamsDataInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ParamsDataInterface);
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
    // Note: CUDA 11.4.2 + GCC 11.2.0 doesn't support 'if constexpr'
    return get_ref<M>(*this);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
