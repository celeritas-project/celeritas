//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CurandPerformance.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Types.hh"

#include <vector>

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
struct TestParams
{
    using size_type = celeritas::size_type;
    using real_type = celeritas::real_type;

    size_type     nsamples;  //! number of samples
    size_type     nthreads;  //! number of threads per blocks
    size_type     nblocks;   //! number of blocks per grids
    unsigned long seed;      //! seed
    real_type     tolerance; //! tolerance of random errors
};

//! Output results
struct TestOutput
{
    std::vector<double> sum;
    std::vector<double> sum2;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
template<class T>
TestOutput curand_test(TestParams params);

#if !CELERITAS_USE_CUDA
template<class T>
inline TestOutput curand_test(TestParams)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
