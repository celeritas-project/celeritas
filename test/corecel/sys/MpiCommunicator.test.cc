//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MpiCommunicator.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/MpiCommunicator.hh"

#include "corecel/cont/Span.hh"
#include "corecel/sys/MpiOperations.hh"
#include "corecel/sys/ScopedMpiInit.hh"

#include "celeritas_test.hh"

#if CELERITAS_USE_MPI
#    define TEST_IF_CELERITAS_MPI(name) name
#else
#    define TEST_IF_CELERITAS_MPI(name) DISABLED_##name
#endif

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(CommunicatorTest, null)
{
    MpiCommunicator comm;
    EXPECT_FALSE(comm);

    // "null" comm always acts like it's running in serial
    EXPECT_EQ(1, comm.size());
    EXPECT_EQ(0, comm.rank());

    // Barrier should be a null-op
    barrier(comm);

    // Reduction should return the original value
    EXPECT_EQ(123, allreduce(comm, Operation::sum, 123));

    // Not-in-place reduction should copy the values
    int const src[] = {1234};
    int dst[] = {-1};
    allreduce(comm, Operation::max, make_span(src), make_span(dst));
    EXPECT_EQ(1234, dst[0]);
}

TEST(CommunicatorTest, TEST_IF_CELERITAS_MPI(self))
{
    MpiCommunicator comm = MpiCommunicator::comm_self();
    EXPECT_NE(MpiCommunicator::comm_world().mpi_comm(), comm.mpi_comm());

    // "self" comm always acts like it's running in serial
    EXPECT_EQ(1, comm.size());
    EXPECT_EQ(0, comm.rank());

    barrier(comm);
    EXPECT_EQ(123, allreduce(comm, Operation::sum, 123));
}

TEST(CommunicatorTest, TEST_IF_CELERITAS_MPI(world))
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
