//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.hh
//---------------------------------------------------------------------------//
#include "base/Macros.hh"
#include "base/StackAllocatorData.hh"

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
    using MockAllocatorData
        = celeritas::StackAllocatorData<MockSecondary,
                                        celeritas::Ownership::reference,
                                        celeritas::MemSpace::device>;

    int               num_threads;
    int               num_iters;
    int               alloc_size;
    MockAllocatorData sa_data;
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

#if !CELER_USE_DEVICE
inline SATestOutput sa_test(const SATestInput&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void sa_clear(const SATestInput&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
