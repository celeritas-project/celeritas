//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file TestMain.cc
//---------------------------------------------------------------------------//
#include "TestMain.hh"

#include <stdexcept>
#include "celeritas_config.h"
#include "base/ColorUtils.hh"
#include "comm/KernelDiagnostics.hh"
#include "celeritas_version.h"
#include "comm/Communicator.hh"
#include "comm/Device.hh"
#include "comm/Logger.hh"
#include "comm/Operations.hh"
#include "comm/ScopedMpiInit.hh"
#include "NonMasterResultPrinter.hh"
#include "ParallelHandler.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
int test_main(int argc, char** argv)
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    Communicator  comm
        = (ScopedMpiInit::status() == ScopedMpiInit::Status::disabled
               ? Communicator{}
               : Communicator::comm_world());

    try
    {
        // Initialize device
        celeritas::activate_device(Device::from_round_robin(comm));
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

    if (comm.rank() == 0)
    {
        std::cout << color_code('x') << "Celeritas version "
                  << celeritas_version << std::endl;
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

    // Find out if any process failed
    failed = allreduce(comm, Operation::max, failed);

    // If no tests were run, there's a problem.
    if (testing::UnitTest::GetInstance()->test_to_run_count() == 0)
    {
        if (comm.rank() == 0)
        {
            std::cout << color_code('r') << "[  FAILED  ]" << color_code(' ')
                      << " No tests are written/enabled!" << std::endl;
        }

        failed = 1;
    }

    if (celeritas::device())
    {
        // Print kernel diagnostics
        std::cout << color_code('x')
                  << "Kernel diagnostics: " << celeritas::kernel_diagnostics()
                  << color_code(' ') << std::endl;
    }

    // Print final results
    if (comm.rank() == 0)
    {
        std::cout << color_code('x') << (argc > 0 ? argv[0] : "UNKNOWN")
                  << ": tests " << (failed ? "FAILED" : "PASSED")
                  << color_code(' ') << std::endl;
    }

    // Return 1 if any failure, 0 if all success
    return failed;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
