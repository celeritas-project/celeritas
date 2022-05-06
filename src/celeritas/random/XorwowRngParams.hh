//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/Types.hh"

#include "XorwowRngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared data for XORWOW pseudo-random number generator.
 */
class XorwowRngParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef
        = XorwowRngParamsData<Ownership::const_reference, MemSpace::host>;
    using DeviceRef
        = XorwowRngParamsData<Ownership::const_reference, MemSpace::device>;
    //!@}

  public:
    // Construct with a low-entropy seed
    explicit XorwowRngParams(unsigned int seed);

    // TODO: Construct with a seed of 256 bytes (16-byte hex) or shasum string
    // explicit XorwowRngParams(const std::string& hexstring);

    //! Access material properties on the host
    const HostRef& host_ref() const { return data_.host(); }

    //! Access material properties on the device
    const DeviceRef& device_ref() const { return data_.device(); }

  private:
    // Host/device storage and reference
    CollectionMirror<XorwowRngParamsData> data_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
