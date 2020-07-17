//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.hh
//---------------------------------------------------------------------------//

#include "base/StackAllocatorPointers.hh"

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct SATestInput
{
    int                num_threads;
    int                num_iters;
    int                alloc_size;
    StackAllocatorPointers sa_view;
};

//---------------------------------------------------------------------------//
//! Output results
struct SATestOutput
{
    int num_allocations;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
SATestOutput sa_run(SATestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
