//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GenerateCanonical.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Generate random numbers in [0, 1).
 */
template<class Generator, class T>
CELER_FUNCTION auto GenerateCanonical<Generator, T>::operator()(Generator& rng)
    -> result_type
{
    return std::generate_canonical<result_type,
                                   std::numeric_limits<result_type>::digits>(
        rng);
}

#ifdef __NVCC__
//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine, float
 */
__device__ float GenerateCanonical<RngEngine, float>::operator()(RngEngine& rng)
{
    return curand_uniform(rng.state());
}

//---------------------------------------------------------------------------//
/*!
 * Specialization for RngEngine, double
 */
__device__ double
GenerateCanonical<RngEngine, double>::operator()(RngEngine& rng)
{
    return curand_uniform_double(rng.state());
}

#endif
//---------------------------------------------------------------------------//
} // namespace celeritas
