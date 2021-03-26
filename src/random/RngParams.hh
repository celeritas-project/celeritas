//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "RngInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Manage random number generation.
 *
 * Currently this just constructs a local seed number but should be extended to
 * handle RNG setup across multiple MPI processes.
 */
class RngParams
{
  public:
    //!@{
    //! References to constructed data
    using HostRef = RngParamsData<Ownership::const_reference, MemSpace::host>;
    //!@}

  public:
    // Construct with seed
    explicit inline RngParams(unsigned int seed);

    //! Access RNG properties for constructing RNG state
    const HostRef& host_pointers() const { return host_ref_; }

  private:
    HostRef host_ref_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITOINS
//---------------------------------------------------------------------------//
/*!
 * Construct with seed.
 */
RngParams::RngParams(unsigned int seed)
{
    host_ref_.seed = seed;
}

} // namespace celeritas
