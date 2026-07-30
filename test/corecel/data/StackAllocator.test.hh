//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/StackAllocator.test.hh
//---------------------------------------------------------------------------//
#include "corecel/Macros.hh"
#include "corecel/data/StackAllocatorData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
struct MockSecondary
{
    int mock_id = -1;  //!< Default to garbage value
};

//! Input data
struct SATestInput
{
    using MockAllocatorData
        = StackAllocatorData<MockSecondary, Ownership::reference, MemSpace::device>;

    size_type num_threads;
    size_type num_iters;
    size_type alloc_size;
    MockAllocatorData sa_data;
};

//---------------------------------------------------------------------------//
//! Output results
struct SATestOutput
{
    size_type num_errors = 0;
    size_type num_allocations = 0;
    size_type view_size = 0;
    ull_int last_secondary_address = 0;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
SATestOutput sa_test(SATestInput const&);
void sa_clear(SATestInput const&);

#if !CELER_USE_DEVICE
inline SATestOutput sa_test(SATestInput const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

inline void sa_clear(SATestInput const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
