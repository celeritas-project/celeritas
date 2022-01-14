//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Logger.test.cc
//---------------------------------------------------------------------------//
#include "comm/Logger.hh"

#include <iomanip>
#include <thread>

#include "celeritas_test.hh"
#include "base/Range.hh"
#include "base/Stopwatch.hh"
#include "comm/Communicator.hh"
#include "comm/Environment.hh"
#include "comm/ScopedMpiInit.hh"

using celeritas::Communicator;
using celeritas::Logger;
using celeritas::LogLevel;
using celeritas::Provenance;
// using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class LoggerTest : public celeritas::Test
{
  protected:
    void SetUp() override
    {
        using celeritas::ScopedMpiInit;
        if (ScopedMpiInit::status() != ScopedMpiInit::Status::disabled)
        {
            comm_self  = Communicator::comm_self();
            comm_world = Communicator::comm_world();
        }
    }

    Communicator comm_self;
    Communicator comm_world;
};

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//

struct ExpensiveToPrint
{
};

std::ostream& operator<<(std::ostream& os, const ExpensiveToPrint&)
{
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2s);
    return os;
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LoggerTest, global_handlers)
{
    CELER_LOG(status) << "This is a status message #" << 1;
    CELER_LOG(warning) << "This is a warning message";
    CELER_LOG(debug) << "This should be hidden by default";
    std::cerr << "[regular cerr]" << std::endl;
    ::celeritas::world_logger().level(LogLevel::debug);
    CELER_LOG(debug) << "This should be shown now";

    CELER_LOG_LOCAL(warning) << "Warning from rank " << comm_world.rank();

    // Replace 'local' with a null-op logger, so the log message will never
    // show
    ::celeritas::self_logger() = Logger(comm_self, nullptr);
    CELER_LOG_LOCAL(critical) << "the last enemy that shall be destroyed is "
                                 "death";
}

TEST_F(LoggerTest, null)
{
    Logger log(comm_self, nullptr);

    log({"<file>", 0}, LogLevel::info) << "This should be fine!";
}

TEST_F(LoggerTest, custom_log)
{
    Provenance  last_prov;
    LogLevel    last_lev = LogLevel::debug;
    std::string last_msg;

    Logger log(comm_self, [&](Provenance prov, LogLevel lev, std::string msg) {
        last_prov = prov;
        last_lev  = lev;
        last_msg  = std::move(msg);
    });

    // Update level
    EXPECT_EQ(LogLevel::status, log.level());
    log.level(LogLevel::warning);
    EXPECT_EQ(LogLevel::warning, log.level());

    // Call (won't be shown)
    log({"file", 0}, LogLevel::info) << "Shouldn't be shown";
    EXPECT_EQ("", last_msg);

    // Call at higher level
    log({"derp", 1}, LogLevel::warning) << "Danger Will Robinson";
    EXPECT_EQ("derp", last_prov.file);
    EXPECT_EQ(1, last_prov.line);
    EXPECT_EQ("Danger Will Robinson", last_msg);

    // Fancy: use local scoping
    {
        auto msg = log({"yo", 2}, LogLevel::error);
        msg << "Things failed because:";
        msg << std::setw(3) << 1;
        msg << " is the loneliest number";
        // Message should not have yet flushed
        EXPECT_EQ(1, last_prov.line);
    }
    // Message should flush
    EXPECT_EQ(2, last_prov.line);
    EXPECT_EQ("Things failed because:  1 is the loneliest number", last_msg);
}

TEST_F(LoggerTest, DISABLED_performance)
{
    // Construct a logger with an expensive output routine that will never be
    // called
    Logger log(comm_self, [&](Provenance prov, LogLevel lev, std::string msg) {
        cout << prov.file << prov.line << static_cast<int>(lev) << msg << endl;
    });
    log.level(LogLevel::critical);

    // Even in debug this takes only 26ms
    celeritas::Stopwatch get_time;
    for (auto i : celeritas::range(100000))
    {
        log({"<file>", 0}, LogLevel::info)
            << "Never printed: " << i << ExpensiveToPrint{};
    }
    EXPECT_GT(0.1, get_time());
}

TEST_F(LoggerTest, env_setup)
{
    auto log_to_cout = [](Provenance prov, LogLevel lev, std::string msg) {
        cout << to_cstring(lev) << ':' << prov.file << ':' << prov.line << ':'
             << msg << endl;
        EXPECT_EQ("This should print", msg);
    };
    auto celer_setenv = [](const std::string& key, const std::string& val) {
        celeritas::environment().insert({key, val});
    };

    {
        celer_setenv("CELER_TEST_ENV_0", "debug");
        Logger log(comm_world, log_to_cout, "CELER_TEST_ENV_0");
        log({"<test>", 0}, LogLevel::debug) << "This should print";
    }
    {
        celer_setenv("CELER_TEST_ENV_1", "error");
        Logger log(comm_world, log_to_cout, "CELER_TEST_ENV_1");
        log({"<test>", 1}, LogLevel::warning) << "This should not";
    }
    {
        CELER_LOG(info) << "We should see a helpful warning below:";
        celer_setenv("CELER_TEST_ENV_2", "TEST_INVALID_VALUE");
        Logger log(comm_world, log_to_cout, "CELER_TEST_ENV_2");
        log({"<test>", 1}, LogLevel::debug) << "Should not print";
    }
}
