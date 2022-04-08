//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <random>

#include "base/device_runtime_api.h"
#include "celeritas_config.h"
#include "base/Assert.hh"
#include "base/Collection.hh"
#include "base/CollectionBuilder.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "comm/Device.hh"

/*!
 * \def CELER_RNG_PREFIX
 *
 * Add a prefix "hip", "cu", or "mock" to a code token. This is used to
 * toggle between the different RNG options.
 */
#if CELERITAS_RNG == CELERITAS_RNG_BUILTIN_ROCRAND
#    include "detail/mockrand.hh"
#    define CELER_RNG_PREFIX(TOK) mock##TOK
#elif CELERITAS_RNG == CELERITAS_RNG_CURAND
// Override an undocumented cuRAND API definition to enable usage in host code.
#    define QUALIFIERS static __forceinline__ __host__ __device__
#    include <curand_kernel.h>
#    define CELER_RNG_PREFIX(TOK) cu##TOK
#elif CELERITAS_RNG == CELERITAS_RNG_HIPRAND
// Override an undocumented hipRAND API definition to enable usage in host
// code.
#    define FQUALIFIERS __forceinline__ __host__ __device__
#    include <hiprand_kernel.h>
#    define CELER_RNG_PREFIX(TOK) hip##TOK
#else
#    error "Invalid selection of CELERITAS_RNG"
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
using mockrandState_t = detail::MockRandState;
#endif

//! RNG state type: curandState_t, hiprandState_t, mockrandState_t
using RngThreadState = CELER_RNG_PREFIX(randState_t);

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
 * Initialize an RNG.
 */
template<MemSpace M>
struct RngInitializer;

template<>
struct RngInitializer<MemSpace::device>
{
    ull_int seed;
};

template<>
struct RngInitializer<MemSpace::host>
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

    StateItems<RngThreadState> rng;

    //// METHODS ////

    //! True if assigned
    explicit CELER_FUNCTION operator bool() const { return !rng.empty(); }

    //! State size
    CELER_FUNCTION size_type size() const { return rng.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    RngStateData& operator=(RngStateData<W2, M2>& other)
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
inline void
resize(RngStateData<Ownership::value, M>*                               state,
       const RngParamsData<Ownership::const_reference, MemSpace::host>& params,
       size_type                                                        size);

} // namespace celeritas
