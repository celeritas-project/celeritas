//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Range.test.hh
//---------------------------------------------------------------------------//
#include "base/Macros.hh"

#include "celeritas_config.h"
#include <cstdint>
#include <vector>

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct RangeTestInput
{
    int              a;
    std::vector<int> x;
    std::vector<int> y;
    unsigned int     num_threads;
    unsigned int     num_blocks;
};

//! Output data
struct RangeTestOutput
{
    std::vector<int> z;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
RangeTestOutput rangedev_test(RangeTestInput);

#if !CELERITAS_USE_CUDA
inline RangeTestOutput rangedev_test(RangeTestInput)
{
    return {};
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
