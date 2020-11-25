//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Communicator.test.cc
//---------------------------------------------------------------------------//
#include "comm/Communicator.hh"

#include "celeritas_test.hh"

using celeritas::barrier;
using celeritas::Communicator;

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

TEST_F(CommunicatorTest, world)
{
    Communicator comm = Communicator::comm_world();

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
}

TEST_F(CommunicatorTest, self)
{
    Communicator comm = Communicator::comm_self();
    EXPECT_NE(Communicator::comm_world().mpi_comm(), comm.mpi_comm());
    barrier(comm);

    // "self" comm always acts like it's running in serial
    EXPECT_EQ(1, comm.size());
    EXPECT_EQ(0, comm.rank());
}
