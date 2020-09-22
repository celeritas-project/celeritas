//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.hh
//---------------------------------------------------------------------------//
#include "base/StackAllocatorPointers.hh"
#include "base/Macros.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
struct MockSecondary
{
    int def_id = -1; //!< Default to garbage value
};

using StackAllocatorPointersMock
    = celeritas::StackAllocatorPointers<MockSecondary>;

//! Input data
struct SATestInput
{
    int                        num_threads;
    int                        num_iters;
    int                        alloc_size;
    StackAllocatorPointersMock sa_pointers;
};

//---------------------------------------------------------------------------//
//! Output results
struct SATestOutput
{
    using ull_int = celeritas::ull_int;

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
