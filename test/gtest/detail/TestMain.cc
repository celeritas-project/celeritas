//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file gtest/detail/TestMain.cc
//---------------------------------------------------------------------------//
#include "TestMain.hh"

#include <stdexcept>

#include "celeritas_config.h"
#include "celeritas_version.h"
#include "corecel/io/ColorUtils.hh"
#include "celeritas/ext/MpiCommunicator.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/KernelDiagnostics.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/ext/MpiOperations.hh"
#include "celeritas/ext/ScopedMpiInit.hh"

#include "NonMasterResultPrinter.hh"
#include "ParallelHandler.hh"

using std::cout;
using std::endl;

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
            cout << color_code('r') << "[  FAILED  ]" << color_code(' ')
                 << " Device failed to initialize: " << e.what() << endl;
        }
        return 1;
    }

    if (comm.rank() == 0)
    {
        cout << color_code('x') << "Celeritas version " << celeritas_version
             << endl;
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
    int  failed   = RUN_ALL_TESTS();
    bool no_tests = testing::UnitTest::GetInstance()->test_to_run_count() == 0;
    failed += (no_tests ? 1 : 0);

    // Find out if any process failed
    failed = allreduce(comm, Operation::max, failed);

    // If no tests were run, there's a problem.
    if (comm.rank() == 0)
    {
        if (no_tests)
        {
            cout << color_code('r') << "[  FAILED  ]" << color_code(' ')
                 << " No tests are written/enabled!" << endl;
        }

        // Write diagnostics and overall test result
        cout << color_code('x');
        if (celeritas::device())
        {
            cout << "Kernel diagnostics: " << celeritas::kernel_diagnostics()
                 << endl;
        }
        cout << "Celeritas environment variables: " << celeritas::environment()
             << endl;

        cout << (argc > 0 ? argv[0] : "UNKNOWN") << ": tests "
             << (failed ? "FAILED" : "PASSED") << color_code(' ') << endl;
    }

    // Return 1 if any failure, 0 if all success
    return failed;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
