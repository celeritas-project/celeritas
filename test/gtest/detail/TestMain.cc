//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TestMain.cc
//---------------------------------------------------------------------------//
#include "TestMain.hh"

#include <stdexcept>
#include "celeritas_config.h"
#include "base/ColorUtils.hh"
#include "comm/Communicator.hh"
#include "comm/ScopedMpiInit.hh"
#include "comm/Utils.hh"
#include "NonMasterResultPrinter.hh"
#include "ParallelHandler.hh"
#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
int test_main(int argc, char** argv)
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    Communicator  comm = Communicator::comm_world();

    // Initialize device
    try
    {
        celeritas::initialize_device(comm);
    }
    catch (const std::exception& e)
    {
        if (comm.rank() == 0)
        {
            std::cout << color_code('r') << "[  FAILED  ]" << color_code(' ')
                      << " CUDA failed to initialize: " << e.what()
                      << std::endl;
        }
        return 1;
    }

    // Initialize google test
    ::testing::InitGoogleTest(&argc, argv);

    // Gets hold of the event listener list.
    ::testing::TestEventListeners& listeners
        = ::testing::UnitTest::GetInstance()->listeners();

    if (comm.rank() != 0)
    {
        // Don't print test completion messages (default pretty printer)
        delete listeners.Release(listeners.default_result_printer());

        // Instead, print just failure message (in case test fails on just one
        // node)
        listeners.Append(new NonMasterResultPrinter(comm.rank()));
    }

    // Adds a listener to the end.  Google Test takes the ownership.
    listeners.Append(new ParallelHandler(comm));

    // Run everything
    int failed = RUN_ALL_TESTS();

    // Accumulate the result so that all processors will have the same result
    // XXX replace with celeritas comm wrappers
    int global_failed = failed;
#if CELERITAS_USE_MPI
    MPI_Allreduce(
        &failed, &global_failed, 1, MPI_INT, MPI_MAX, comm.mpi_comm());
#endif

    // If no tests were run, there's a problem.
    if (testing::UnitTest::GetInstance()->test_to_run_count() == 0)
    {
        if (comm.rank() == 0)
        {
            std::cout << color_code('r') << "[  FAILED  ]" << color_code(' ')
                      << " No tests are written/enabled!" << std::endl;
        }

        global_failed = 1;
    }

    // Print final results
    if (comm.rank() == 0)
    {
        std::cout << color_code('x') << (argc > 0 ? argv[0] : "UNKNOWN")
                  << ": tests " << (global_failed ? "FAILED" : "PASSED")
                  << color_code(' ') << std::endl;
    }

    // Return 1 if any failure, 0 if all success
    return global_failed;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
