//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.test.cc
//---------------------------------------------------------------------------//
#include "base/StackAllocatorStore.hh"

#include <memory>
#include "gtest/Main.hh"
#include "gtest/Test.hh"
#include "StackAllocator.test.hh"

using celeritas::StackAllocatorStore;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class StackAllocatorTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        // Allocate 8 kiB of device stack space
        storage = StackAllocatorStore(1024 * 8);
    }
    StackAllocatorStore storage;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(StackAllocatorTest, all)
{
    EXPECT_EQ(1024 * 8, storage.capacity());

    // Allocate a subset of the stack
    SATestInput input;
    input.sa_view     = storage.device_view();
    input.num_threads = 256;
    input.num_iters   = 1;
    input.alloc_size  = 8;
    auto result       = sa_run(input);
    EXPECT_EQ(256, result.num_allocations);

    // Overflow the stack
    input.num_iters = 16;
    result          = sa_run(input);
    EXPECT_EQ(1024 - 256, result.num_allocations);

    // Swap with another (smaller) stack
    StackAllocatorStore other(128 * 8);
    storage.swap(other);
    input.sa_view = storage.device_view();
    EXPECT_EQ(128 * 8, storage.capacity());
    EXPECT_EQ(1024 * 8, other.capacity());

    // Run a lot of threads with a small allocation
    input.alloc_size  = 1;
    input.num_iters   = 8;
    input.num_threads = 1024;
    result            = sa_run(input);
    EXPECT_EQ(128 * 8, result.num_allocations);

    // Clear allocated data and run with very few threads
    storage.clear();
    input.alloc_size  = 16;
    input.num_threads = 8;
    result            = sa_run(input);
    EXPECT_EQ(128 / 2, result.num_allocations);
}
