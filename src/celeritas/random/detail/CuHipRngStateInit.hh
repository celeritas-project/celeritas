//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/detail/CuHipRngStateInit.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"

#include "../CuHipRngData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct CuHipRngInitData
{
    StateCollection<ull_int, W, M> seeds;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !seeds.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return seeds.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CuHipRngInitData& operator=(CuHipRngInitData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        seeds = other.seeds;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Initialize the RNG state on host/device
void rng_state_init(DeviceCRef<CuHipRngParamsData> const& params,
                    DeviceRef<CuHipRngStateData> const& state,
                    DeviceCRef<CuHipRngInitData> const& seeds);

void rng_state_init(HostCRef<CuHipRngParamsData> const& params,
                    HostRef<CuHipRngStateData> const& state,
                    HostCRef<CuHipRngInitData> const& seeds);

#if !CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
inline void rng_state_init(DeviceCRef<CuHipRngParamsData> const&,
                           DeviceRef<CuHipRngStateData> const&,
                           DeviceCRef<CuHipRngInitData> const&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
