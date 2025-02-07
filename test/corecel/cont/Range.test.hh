//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Range.test.hh
//---------------------------------------------------------------------------//
#include <cstdint>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/Macros.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct RangeTestInput
{
    int a;
    std::vector<int> x;
    std::vector<int> y;
    unsigned int num_threads;
};

//! Output data
struct RangeTestOutput
{
    std::vector<int> z;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
RangeTestOutput rangedev_test(RangeTestInput);

#if !CELER_USE_DEVICE
inline RangeTestOutput rangedev_test(RangeTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
