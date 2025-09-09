//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/data/CuHipRngData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <random>

#include "corecel/Config.hh"
#include "corecel/DeviceRuntimeApi.hh"

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
#    if (HIP_VERSION_MAJOR > 5 \
         || (HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR >= 3))
// Override an undocumented hipRAND API definition to enable usage in host
// code.
#        define QUALIFIERS __forceinline__ __host__ __device__
#    else
// Override an older version of that macro
#        define FQUALIFIERS __forceinline__ __host__ __device__
#    endif
#    pragma clang diagnostic push
// "Disabled inline asm, because the build target does not support it."
#    pragma clang diagnostic ignored "-W#warnings"
// "ignoring return value of function declared with 'nodiscard' attribute"
#    pragma clang diagnostic ignored "-Wunused-result"
#    include <hiprand/hiprand_kernel.h>
#    pragma clang diagnostic pop
#    define CELER_RNG_PREFIX(TOK) hip##TOK
#else
// CuHipRng is invalid
#    include "detail/MockRand.hh"

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
 */
template<Ownership W, MemSpace M>
struct CuHipRngParamsData
{
    //// DATA ////

    unsigned int seed = 12345;

    //// METHODS ////

    //! Any settings are valid
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    CuHipRngParamsData& operator=(CuHipRngParamsData<W2, M2> const& other)
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
    ull_int seed{0};
    ull_int subsequence{0};
    ull_int offset{0};
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
            HostCRef<CuHipRngParamsData> const& params,
            StreamId stream,
            size_type size);

}  // namespace celeritas
