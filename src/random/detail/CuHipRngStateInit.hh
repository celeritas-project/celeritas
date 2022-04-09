//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CuHipRngStateInit.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Assert.hh"
#include "base/Collection.hh"

#include "../CuHipRngData.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<Ownership W, MemSpace M>
struct CuHipRngInitData
{
    StateCollection<CuHipRngInitializer, W, M> seeds;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !seeds.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return seeds.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CuHipRngInitData& operator=(const CuHipRngInitData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        seeds = other.seeds;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Initialize the RNG state on host/device
void rng_state_init(
    const CuHipRngStateData<Ownership::reference, MemSpace::device>&      rng,
    const CuHipRngInitData<Ownership::const_reference, MemSpace::device>& seeds);

void rng_state_init(
    const CuHipRngStateData<Ownership::reference, MemSpace::host>&      rng,
    const CuHipRngInitData<Ownership::const_reference, MemSpace::host>& seeds);

#if !CELER_USE_DEVICE
//---------------------------------------------------------------------------//
/*!
 * Initialize the RNG states on device from seeds randomly generated on host.
 */
inline void rng_state_init(
    const CuHipRngStateData<Ownership::reference, MemSpace::device>&,
    const CuHipRngInitData<Ownership::const_reference, MemSpace::device>&)
{
    CELER_ASSERT_UNREACHABLE();
}
#endif

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
