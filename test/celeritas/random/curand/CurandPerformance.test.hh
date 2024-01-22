//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/curand/CurandPerformance.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Types.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
struct TestParams
{
    size_type nsamples;  //! number of samples
    size_type nthreads;  //! number of threads per blocks
    size_type nblocks;  //! number of blocks per grids
    unsigned long seed;  //! seed
    real_type tolerance;  //! tolerance of random errors
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
