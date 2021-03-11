//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.hh
//---------------------------------------------------------------------------//
#include "base/StackAllocatorInterface.hh"
#include "base/Macros.hh"

namespace celeritas_test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
struct MockSecondary
{
    int mock_id = -1; //!< Default to garbage value
};

//! Input data
struct SATestInput
{
    using MockAllocatorPointers
        = celeritas::StackAllocatorData<MockSecondary,
                                        celeritas::Ownership::reference,
                                        celeritas::MemSpace::device>;

    int                   num_threads;
    int                   num_iters;
    int                   alloc_size;
    MockAllocatorPointers sa_pointers;
};

//---------------------------------------------------------------------------//
//! Output results
struct SATestOutput
{
    using ull_int = celeritas::ull_int;

    int     num_errors             = 0;
    int     num_allocations        = 0;
    int     view_size              = 0;
    ull_int last_secondary_address = 0;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
SATestOutput sa_test(const SATestInput&);
void         sa_clear(const SATestInput&);

#if !CELERITAS_USE_CUDA
inline SATestOutput sa_test(const SATestInput&)
{
    CELER_NOT_CONFIGURED("CUDA");
}

inline void sa_clear(const SATestInput&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
