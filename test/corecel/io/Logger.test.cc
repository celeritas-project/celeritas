//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Logger.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/Logger.hh"

#include <iomanip>
#include <thread>

#include "corecel/cont/Range.hh"
#include "corecel/io/detail/NullLoggerMessage.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "corecel/sys/ScopedMpiInit.hh"
#include "corecel/sys/Stopwatch.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
TEST(NullLoggerMessage, all)
{
    NullLoggerMessage() << "This should not print anything " << range(10)
                        << std::setw(5) << 123;
}

TEST(LoggerTypes, all)
{
    EXPECT_TRUE(std::is_trivially_destructible_v<LogProvenance>);
}

}  // namespace test
}  // namespace detail

namespace test
{

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class LoggerTest : public Test
{
  protected:
    void SetUp() override
    {
        if (ScopedMpiInit::status() != ScopedMpiInit::Status::disabled)
        {
            comm_self = MpiCommunicator::self();
            comm_world = MpiCommunicator::world();
        }
    }

    MpiCommunicator comm_self;
    MpiCommunicator comm_world;
};

//---------------------------------------------------------------------------//
// HELPER CLASSES
//---------------------------------------------------------------------------//

struct ExpensiveToPrint
{
};

std::ostream& operator<<(std::ostream& os, ExpensiveToPrint const&)
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
    world_logger().level(LogLevel::debug);
    CELER_LOG(debug) << "This should be shown now";

    CELER_LOG_LOCAL(warning) << "Warning from rank " << comm_world.rank();

    // Replace 'local' with a null-op logger, so the log message will never
    // show
    self_logger() = Logger(nullptr);
    CELER_LOG_LOCAL(critical)
        << R"(It is pitch black. You are likely to be eaten by a grue.)";
}

TEST_F(LoggerTest, null)
{
    Logger log(nullptr);

    log({"<file>", 0}, LogLevel::info) << "This should be fine!";
}

TEST_F(LoggerTest, custom_log)
{
    LogProvenance last_prov;
    LogLevel last_lev = LogLevel::debug;
    std::string last_msg;

    Logger log([&](LogProvenance prov, LogLevel lev, std::string msg) {
        last_prov = prov;
        last_lev = lev;
        last_msg = std::move(msg);
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
    Logger log([&](LogProvenance prov, LogLevel lev, std::string msg) {
        cout << prov.file << prov.line << static_cast<int>(lev) << msg << endl;
    });
    log.level(LogLevel::critical);

    // Even in debug this takes only 26ms
    Stopwatch get_time;
    for (auto i : range(100000))
    {
        log({"<file>", 0}, LogLevel::info)
            << "Never printed: " << i << ExpensiveToPrint{};
    }
    EXPECT_GT(0.1, get_time());
}

TEST_F(LoggerTest, level_from_env)
{
    auto set_level = [](std::string const& key, std::string const& val) {
        environment().insert({key, val});
        return log_level_from_env(key);
    };

    EXPECT_EQ(LogLevel::debug, set_level("CELER_TEST_ENV_0", "debug"));
    EXPECT_EQ(LogLevel::error, set_level("CELER_TEST_ENV_1", "error"));
    EXPECT_THROW(set_level("CELER_TEST_ENV_2", "not_a_log_level"),
                 RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
