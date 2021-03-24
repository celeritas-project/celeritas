//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#if CELERITAS_USE_CUDA
/*!
 * \def QUALIFIERS
 *
 * Override an undocumented CURAND API definition to enable usage in host code.
 */
#    define QUALIFIERS static __forceinline__ __host__ __device__
#    include <curand_kernel.h>
#else
#    include "detail/curand.nocuda.hh"
#endif

#include "base/Collection.hh"
#include "base/Types.hh"

#if !CELERITAS_USE_CUDA
//! Define an unused RNG state for "device" code to support no-cuda build
using curandState_t = celeritas::detail::MockCurandState;
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Properties of the global random number generator.
 *
 * There is no persistent data needed on device or at runtime: the params are
 * only used for construction.
 */
template<Ownership W, MemSpace M>
struct RngParamsData;

template<Ownership W>
struct RngParamsData<W, MemSpace::device>
{
    /* no data on device */

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RngParamsData& operator=(const RngParamsData<W2, M2>&)
    {
        return *this;
    }
};

template<Ownership W>
struct RngParamsData<W, MemSpace::host>
{
    //// DATA ////

    unsigned int seed = 12345; // TODO: replace with std::seed_seq etc

    //// METHODS ////

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RngParamsData& operator=(const RngParamsData<W2, M2>& other)
    {
        seed = other.seed;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * The underlying RNG state is *different* on host and device.
 */
template<MemSpace M>
struct RngThreadState;

template<>
struct RngThreadState<MemSpace::device>
{
    curandState_t state;
};

template<>
struct RngThreadState<MemSpace::host>
{
    // TODO: std::shared_ptr<std::mt19937> rng; ?
};

//---------------------------------------------------------------------------//
/*!
 * Initialize an RNG.
 */
template<MemSpace M>
struct RngIntializer;

template<>
struct RngIntializer<MemSpace::device>
{
    ull_int seed;
};

//---------------------------------------------------------------------------//
/*!
 * RNG state data.
 */
template<Ownership W, MemSpace M>
struct RngStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StateItems<RngThreadState<M>> rng;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !rng.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return rng.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RngStateData& operator=(RngStateData<W2, M2>& other)
    {
        static_assert(M == M2,
                      "RNG state cannot be transferred between host and "
                      "device because they use separate RNG types");
        CELER_EXPECT(other);
        rng = other.rng;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// Resize and initialize with the seed stored in params.
void resize(
    RngStateData<Ownership::value, MemSpace::device>*                state,
    const RngParamsData<Ownership::const_reference, MemSpace::host>& params,
    size_type                                                        size);

//---------------------------------------------------------------------------//
} // namespace celeritas
