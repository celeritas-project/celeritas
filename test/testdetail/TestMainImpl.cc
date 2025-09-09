//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file testdetail/TestMainImpl.cc
//---------------------------------------------------------------------------//
#include "TestMainImpl.hh"

#include <stdexcept>
#include <string_view>

#include "corecel/Config.hh"
#include "corecel/Version.hh"

#include "corecel/Macros.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/MpiOperations.hh"
#include "corecel/sys/ScopedMpiInit.hh"

#include "NonMasterResultPrinter.hh"

using std::cout;
using std::endl;

namespace celeritas
{
namespace testdetail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Add barriers at test ends.
 */
class ParallelListener final : public ::testing::EmptyTestEventListener
{
  public:
    //! Construct with communicator
    explicit ParallelListener(MpiCommunicator const& comm) : comm_(comm) {}

    //! Write parallel testing info at startup
    void OnTestProgramStart(::testing::UnitTest const&) override
    {
        if (comm_.rank() == 0)
        {
            std::cout << color_code('x') << "Testing "
                      << "on " << comm_.size() << " process"
                      << (comm_.size() > 1 ? "es" : "") << color_code(' ')
                      << std::endl;
        }
    }

    //! Barrier at beginning
    void OnTestStart(::testing::TestInfo const&) override { barrier(comm_); }

    //! Flush and barrier at end
    void OnTestEnd(::testing::TestInfo const&) override
    {
        std::cout << std::flush;
        barrier(comm_);
    }

  private:
    MpiCommunicator const& comm_;
};

//---------------------------------------------------------------------------//
/*!
 * Add barriers at test ends.
 */
class DeviceSkipper final : public ::testing::EmptyTestEventListener
{
  public:
    void OnTestStart(::testing::TestInfo const& test_info) override
    {
        constexpr auto npos = std::string_view::npos;
        for (char const* s : {test_info.test_case_name(), test_info.name()})
        {
            std::string_view sview{s};
            if (sview.find("Device") != npos || sview.find("device") != npos)
            {
                GTEST_SKIP() << "Skipping device test";
            }
        }
    }
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
int test_main(int argc, char** argv)
{
    ScopedMpiInit scoped_mpi(&argc, &argv);
    auto const& comm = celeritas::comm_world();

    try
    {
        // Initialize device
        celeritas::activate_device(comm);
    }
    catch (std::exception const& e)
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
        cout << color_code('x') << "Celeritas version " << version_string
             << color_code(' ') << endl;
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

    if (CELER_USE_DEVICE && !celeritas::device())
    {
        cout << color_code('y') << "Disabling tests with 'device' in the name"
             << color_code(' ') << endl;
        // Skip test parts if "device" in name and device isn't available
        listeners.Append(new DeviceSkipper());
    }

    if (CELERITAS_USE_MPI && comm)
    {
        // Add MPI barriers at test end: Google Test takes ownership of pointer
        listeners.Append(new ParallelListener(comm));
    }

    // Run everything
    int failed = RUN_ALL_TESTS();

    // Check that all processes ran at least one test
    bool no_tests = testing::UnitTest::GetInstance()->test_to_run_count() == 0;
    failed += (no_tests ? 1 : 0);
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
        CELER_LOG(debug) << "Celeritas environment variables: "
                         << environment();

        cout << color_code('x') << (argc > 0 ? argv[0] : "UNKNOWN")
             << ": tests " << (failed ? "FAILED" : "PASSED") << color_code(' ')
             << endl;
    }

    if (auto& d = celeritas::device())
    {
        // Destroy device before exiting
        d.destroy_streams();
    }

    // Return 1 if any failure, 0 if all success
    return failed;
}

//---------------------------------------------------------------------------//
}  // namespace testdetail
}  // namespace celeritas
