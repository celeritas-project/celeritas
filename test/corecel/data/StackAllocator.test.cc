//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/StackAllocator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/StackAllocator.hh"

#include <cstdint>

#include "corecel/data/StateDataStore.hh"

#include "StackAllocator.test.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

template<Ownership W, MemSpace M>
using MockAllocatorData = StackAllocatorData<MockSecondary, W, M>;

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class StackAllocatorTest : public Test
{
  protected:
    using Allocator = StackAllocator<MockSecondary>;

    // Get the actual number of allocated secondaries
    int actual_allocations(SATestInput const& in, SATestOutput const& out) const
    {
        using std::uintptr_t;

        // Use pointer arithmetic on GPUs:
        // - "start" pointer of final allocation
        // - Plus size of final allocation
        // - Minus allocation "begin" address
        MockSecondary* storage_end_ptr
            = static_cast<MockSecondary*>(
                  reinterpret_cast<void*>(out.last_secondary_address))
              + in.alloc_size;
        constexpr ItemId<MockSecondary> first_item{0};
        return storage_end_ptr - &in.sa_data.storage[first_item];
    }
};

//---------------------------------------------------------------------------//

TEST_F(StackAllocatorTest, host)
{
    using StateStore = StateDataStore<MockAllocatorData, MemSpace::host>;
    StateStore data(16);
    Allocator alloc(data.ref());
    EXPECT_EQ(16, alloc.capacity());

    // Allocate 8 of the 16 slots
    MockSecondary* ptr = alloc(8);
    EXPECT_EQ(8, alloc.get().size());
    ASSERT_NE(nullptr, ptr);
    for (MockSecondary& p : Span<MockSecondary>(ptr, 8))
    {
        // Check that secondary was initialized properly
        EXPECT_EQ(-1, p.mock_id);

        p.mock_id = 1;
    }

    // Ask for one more than we have room
    ptr = alloc(9);
    EXPECT_EQ(nullptr, ptr);
    EXPECT_EQ(8, alloc.get().size());

    // Ask for an amount that barely fits
    ptr = alloc(8);
    ASSERT_NE(nullptr, ptr);
    EXPECT_EQ(16, alloc.get().size());
    EXPECT_EQ(16, const_cast<Allocator const&>(alloc).get().size());
}

//---------------------------------------------------------------------------//

TEST_F(StackAllocatorTest, TEST_IF_CELER_DEVICE(device))
{
    using StateStore = StateDataStore<MockAllocatorData, MemSpace::device>;

    StateStore data(1024);

    EXPECT_EQ(1024, data.ref().capacity());
    int accum_expected_alloc = 0;

    // Allocate a subset of the stack
    SATestInput input;
    input.sa_data = data.ref();
    input.num_threads = 64;
    input.num_iters = 1;
    input.alloc_size = 2;
    auto result = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(input.num_threads * input.num_iters * input.alloc_size,
              result.num_allocations);
    accum_expected_alloc += result.num_allocations;
    EXPECT_EQ(accum_expected_alloc, actual_allocations(input, result));
    EXPECT_EQ(accum_expected_alloc, result.view_size);

    // Run again, two iterations per thread
    input.num_iters = 2;
    input.alloc_size = 4;
    result = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(input.num_threads * input.num_iters * input.alloc_size,
              result.num_allocations);
    accum_expected_alloc += result.num_allocations;
    EXPECT_EQ(accum_expected_alloc, actual_allocations(input, result));
    EXPECT_EQ(accum_expected_alloc, result.view_size);

    // Run again, too many iterations (so storage gets filled up)
    input.num_iters = 128;
    input.num_threads = 1024;
    input.alloc_size = 1;
    result = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(1024 - accum_expected_alloc, result.num_allocations);
    EXPECT_EQ(1024, actual_allocations(input, result));
    EXPECT_EQ(1024, result.view_size);

    // Reset secondary storage
    sa_clear(input);
    EXPECT_EQ(1024, data.ref().capacity());

    // Run again until full
    input.num_threads = 512;
    input.num_iters = 3;
    input.alloc_size = 4;
    result = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(1024, result.num_allocations);
    EXPECT_EQ(1024, actual_allocations(input, result));
    EXPECT_EQ(1024, result.view_size);
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
