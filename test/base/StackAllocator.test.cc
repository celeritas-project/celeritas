//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.cc
//---------------------------------------------------------------------------//
#include "base/StackAllocatorStore.hh"
#include "base/StackAllocatorView.hh"

#include <cstdint>
#include "celeritas_test.hh"
#include "StackAllocator.test.hh"
#include "HostStackAllocatorStore.hh"
#include "base/StackAllocatorStore.t.hh"

using namespace celeritas_test;

// Explicitly instantiate mock stack allocator
namespace celeritas
{
template class StackAllocatorStore<MockSecondary>;
}

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class StackAllocatorHostTest : public celeritas::Test
{
  protected:
    using StackAllocatorView = celeritas::StackAllocatorView<MockSecondary>;

    HostStackAllocatorStore<MockSecondary> secondaries_;
};

TEST_F(StackAllocatorHostTest, allocation)
{
    // Set "uninitialized" storage to invalid value
    secondaries_.resize(16, MockSecondary{-123});
    StackAllocatorView alloc(secondaries_.host_pointers());
    EXPECT_EQ(16, alloc.capacity());

    // Allocate 8 of the 16 slots
    MockSecondary* ptr = alloc(8);
    EXPECT_EQ(8, alloc.get().size());
    ASSERT_NE(nullptr, ptr);
    for (MockSecondary& p : celeritas::span<MockSecondary>(ptr, 8))
    {
        // Check that secondary was initialized properly
        EXPECT_EQ(-1, p.def_id);

        p.def_id = 1;
    }

    // Ask for one more than we have room
    ptr = alloc(9);
    EXPECT_EQ(nullptr, ptr);
    EXPECT_EQ(8, alloc.get().size());

    // Ask for an amount that barely fits
    ptr = alloc(8);
    ASSERT_NE(nullptr, ptr);
    EXPECT_EQ(16, alloc.get().size());
    EXPECT_EQ(16, const_cast<const StackAllocatorView&>(alloc).get().size());
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//
#if CELERITAS_USE_CUDA

class StackAllocatorDeviceTest : public celeritas::Test
{
  protected:
    using StackAllocatorStore = celeritas::StackAllocatorStore<MockSecondary>;

    void SetUp() override
    {
        // Allocate 1024 secondaries
        storage = StackAllocatorStore(1024);
    }

    // Get the actual number of allocated secondaries
    int actual_allocations(const SATestInput& in, const SATestOutput& out) const
    {
        using std::uintptr_t;

        // Use GPU pointer arithmetic:
        // - "start" pointer of final allocation
        // - Plus size of final allocation
        // - Minus allocation "begin" address
        MockSecondary* storage_end_ptr
            = static_cast<MockSecondary*>(
                  reinterpret_cast<void*>(out.last_secondary_address))
              + in.alloc_size;
        return storage_end_ptr - in.sa_pointers.storage.data();
    }

    StackAllocatorStore storage;
};

TEST_F(StackAllocatorDeviceTest, run)
{
    EXPECT_EQ(1024, storage.capacity());
    int accum_expected_alloc = 0;

    // Allocate a subset of the stack
    SATestInput input;
    input.sa_pointers = storage.device_pointers();
    input.num_threads = 64;
    input.num_iters   = 1;
    input.alloc_size  = 2;
    auto result       = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(input.num_threads * input.num_iters * input.alloc_size,
              result.num_allocations);
    accum_expected_alloc += result.num_allocations;
    EXPECT_EQ(accum_expected_alloc, actual_allocations(input, result));
    EXPECT_EQ(accum_expected_alloc, result.max_size);
    EXPECT_EQ(accum_expected_alloc, result.view_size);
    EXPECT_EQ(accum_expected_alloc, storage.get_size());

    // Run again, two iterations per thread
    input.num_iters  = 2;
    input.alloc_size = 4;
    result           = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(input.num_threads * input.num_iters * input.alloc_size,
              result.num_allocations);
    accum_expected_alloc += result.num_allocations;
    EXPECT_EQ(accum_expected_alloc, actual_allocations(input, result));
    EXPECT_EQ(accum_expected_alloc, result.max_size);
    EXPECT_EQ(accum_expected_alloc, result.view_size);
    EXPECT_EQ(accum_expected_alloc, storage.get_size());

    // Run again, too many iterations (so storage gets filled up)
    input.num_iters   = 128;
    input.num_threads = 1024;
    input.alloc_size  = 1;
    result            = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(1024 - accum_expected_alloc, result.num_allocations);
    EXPECT_EQ(1024, actual_allocations(input, result));
    EXPECT_LE(1024, result.max_size);
    EXPECT_EQ(1024, result.view_size);
    EXPECT_EQ(1024, storage.get_size());

    // Reset secondary storage
    storage.clear();
    EXPECT_EQ(1024, storage.capacity());
    EXPECT_EQ(0, storage.get_size());

    // Run again until full
    input.num_threads = 512;
    input.num_iters   = 3;
    input.alloc_size  = 4;
    result            = sa_test(input);
    EXPECT_EQ(0, result.num_errors);
    EXPECT_EQ(1024, result.num_allocations);
    EXPECT_EQ(1024, actual_allocations(input, result));
    EXPECT_LE(1024, result.max_size);
    EXPECT_EQ(1024, result.view_size);
    EXPECT_EQ(1024, storage.get_size());

    // Check move operation
    {
        StackAllocatorStore temp_store = std::move(storage);
        EXPECT_EQ(1024, temp_store.capacity());
        EXPECT_EQ(1024, temp_store.get_size());
    }
}

#endif
