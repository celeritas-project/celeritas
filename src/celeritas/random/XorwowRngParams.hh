//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "XorwowRngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Shared data for XORWOW pseudo-random number generator.
 */
class XorwowRngParams final : public ParamsDataInterface<XorwowRngParamsData>
{
  public:
    // Construct with a low-entropy seed
    explicit XorwowRngParams(unsigned int seed);

    // TODO: Construct with a seed of 256 bytes (16-byte hex) or shasum string
    // explicit XorwowRngParams(const std::string& hexstring);

    //! Access material properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access material properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<XorwowRngParamsData> data_;

    //// TYPES ////

    using JumpPoly = Array<unsigned int, 5>;
    using ArrayJumpPoly = Array<JumpPoly, 32>;

    //// HELPER FUNCTIONS ////

    ArrayJumpPoly const& get_jump_poly();
    ArrayJumpPoly const& get_jump_subsequence_poly();
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
