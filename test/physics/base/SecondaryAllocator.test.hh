//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryAllocator.test.hh
//---------------------------------------------------------------------------//
#include "physics/base/SecondaryAllocatorPointers.hh"

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct SATestInput
{
    int                        num_threads;
    int                        num_iters;
    int                        alloc_size;
    SecondaryAllocatorPointers sa_view;
};

//---------------------------------------------------------------------------//
//! Output results
struct SATestOutput
{
    using ull_int = unsigned long long int;

    int     num_errors             = 0;
    int     num_allocations        = 0;
    int     max_size               = 0;
    int     view_size              = 0;
    ull_int last_secondary_address = 0;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
SATestOutput sa_test(SATestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
