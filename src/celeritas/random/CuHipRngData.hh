//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/CuHipRngData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <random>

#include "celeritas_config.h"
#include "corecel/device_runtime_api.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/sys/Device.hh"

/*!
 * \def CELER_RNG_PREFIX
 *
 * Add a prefix "hip", "cu", or "mock" to a code token. This is used to
 * toggle between the different RNG options.
 */
#if CELERITAS_USE_CUDA
// Override an undocumented cuRAND API definition to enable usage in host code.
#    define QUALIFIERS static __forceinline__ __host__ __device__
#    include <curand_kernel.h>
#    define CELER_RNG_PREFIX(TOK) cu##TOK
#elif CELERITAS_USE_HIP
// Override an undocumented hipRAND API definition to enable usage in host
// code.
#    define FQUALIFIERS __forceinline__ __host__ __device__
#    include <hiprand/hiprand_kernel.h>
#    define CELER_RNG_PREFIX(TOK) hip##TOK
#else
// CuHipRng is invalid
#    include "detail/mockrand.hh"
#    define CELER_RNG_PREFIX(TOK) mock##TOK
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
using mockrandState_t = detail::MockRandState;
#endif

//! RNG state type: curandState_t, hiprandState_t, mockrandState_t
using CuHipRngThreadState = CELER_RNG_PREFIX(randState_t);

//---------------------------------------------------------------------------//
/*!
 * Properties of the global random number generator.
 *
 * There is no persistent data needed on device or at runtime: the params are
 * only used for construction.
 */
template<Ownership W, MemSpace M>
struct CuHipRngParamsData;

template<Ownership W>
struct CuHipRngParamsData<W, MemSpace::device>
{
    /* no data on device */

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CuHipRngParamsData& operator=(const CuHipRngParamsData<W2, M2>&)
    {
        return *this;
    }
};

template<Ownership W>
struct CuHipRngParamsData<W, MemSpace::host>
{
    //// DATA ////

    unsigned int seed = 12345;

    //// METHODS ////

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CuHipRngParamsData& operator=(const CuHipRngParamsData<W2, M2>& other)
    {
        seed = other.seed;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Initialize an RNG.
 */
struct CuHipRngInitializer
{
    ull_int seed;
};

//---------------------------------------------------------------------------//
/*!
 * RNG state data.
 */
template<Ownership W, MemSpace M>
struct CuHipRngStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = StateCollection<T, W, M>;

    //// DATA ////

    StateItems<CuHipRngThreadState> rng;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !rng.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return rng.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CuHipRngStateData& operator=(CuHipRngStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        rng = other.rng;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize and initialize with the seed stored in params.
 */
template<MemSpace M>
void resize(CuHipRngStateData<Ownership::value, M>* state,
            const HostCRef<CuHipRngParamsData>&     params,
            size_type                               size);

} // namespace celeritas
