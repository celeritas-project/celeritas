//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/MpiCommunicator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Span.hh"
#include "celeritas/ext/MpiCommunicator.hh"
#include "celeritas/ext/MpiOperations.hh"
#include "celeritas/ext/ScopedMpiInit.hh"

#include "celeritas_test.hh"

using celeritas::MpiCommunicator;
using celeritas::Operation;
using celeritas::ScopedMpiInit;

#if CELERITAS_USE_MPI
#    define TEST_IF_CELERITAS_MPI(name) name
#else
#    define TEST_IF_CELERITAS_MPI(name) DISABLED_##name
#endif

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class CommunicatorTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(CommunicatorTest, null)
{
    using celeritas::make_span;

    MpiCommunicator comm;
    EXPECT_FALSE(comm);

    // "null" comm always acts like it's running in serial
    EXPECT_EQ(1, comm.size());
    EXPECT_EQ(0, comm.rank());

    // Barrier should be a null-op
    celeritas::barrier(comm);

    // Reduction should return the original value
    EXPECT_EQ(123, celeritas::allreduce(comm, Operation::sum, 123));

    // Not-in-place reduction should copy the values
    const int src[] = {1234};
    int       dst[] = {-1};
    celeritas::allreduce(comm, Operation::max, make_span(src), make_span(dst));
    EXPECT_EQ(1234, dst[0]);
}

TEST_F(CommunicatorTest, TEST_IF_CELERITAS_MPI(self))
{
    MpiCommunicator comm = MpiCommunicator::comm_self();
    EXPECT_NE(MpiCommunicator::comm_world().mpi_comm(), comm.mpi_comm());

    // "self" comm always acts like it's running in serial
    EXPECT_EQ(1, comm.size());
    EXPECT_EQ(0, comm.rank());

    barrier(comm);
    EXPECT_EQ(123, celeritas::allreduce(comm, Operation::sum, 123));
}

TEST_F(CommunicatorTest, TEST_IF_CELERITAS_MPI(world))
{
    MpiCommunicator comm = MpiCommunicator::comm_world();

#if CELERITAS_USE_MPI
    EXPECT_EQ(MPI_COMM_WORLD, comm.mpi_comm());

    // Test MPI-specific functionality
    int expected_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &expected_rank);
    EXPECT_EQ(expected_rank, comm.rank());

    int expected_size;
    MPI_Comm_size(MPI_COMM_WORLD, &expected_size);
    EXPECT_EQ(expected_size, comm.size());
#endif

    barrier(comm);

    EXPECT_EQ(123 * comm.size(), allreduce(comm, Operation::sum, 123));
}
