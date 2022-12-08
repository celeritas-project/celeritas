//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/MultiExceptionHandler.hh"

#include <regex>

#include "corecel/io/Logger.hh"
#include "corecel/sys/MpiCommunicator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class MultiExceptionHandlerTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override
    {
        using namespace std::placeholders;
        celeritas::self_logger() = Logger(
            MpiCommunicator{},
            std::bind(
                &MultiExceptionHandlerTest::log_message, this, _1, _2, _3));
    }

    void log_message(Provenance, LogLevel lev, std::string msg)
    {
        EXPECT_EQ(LogLevel::critical, lev);

        static const std::regex delete_ansi("\033\\[[0-9;]*m");
        messages.push_back(std::regex_replace(msg, delete_ansi, ""));
    }

    static void TearDownTestCase()
    {
        // Restore logger
        celeritas::self_logger() = celeritas::make_default_self_logger();
    }

    std::vector<std::string> messages;
};

TEST_F(MultiExceptionHandlerTest, single)
{
    MultiExceptionHandler capture_exception;
    EXPECT_TRUE(capture_exception.empty());
    CELER_TRY_ELSE(
        throw RuntimeError::from_validate("first exception", "", "here", 1),
        capture_exception);
    EXPECT_FALSE(capture_exception.empty());

    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);
}

TEST_F(MultiExceptionHandlerTest, multi)
{
    MultiExceptionHandler capture_exception;
    CELER_TRY_ELSE(
        throw RuntimeError::from_validate("first exception", "", "here", 1),
        capture_exception);
    for (auto i : range(3))
    {
        DebugErrorDetails deets{
            DebugErrorType::internal, "false", "test.cc", i};
        CELER_TRY_ELSE(throw DebugError(deets), capture_exception);
    }
    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);

    static const std::string expected_messages[]
        = {"ignoring exception: test.cc:0:\nceleritas: internal assertion "
           "failed: false",
           "ignoring exception: test.cc:1:\nceleritas: internal assertion "
           "failed: false",
           "ignoring exception: test.cc:2:\nceleritas: internal assertion "
           "failed: false"};
    EXPECT_VEC_EQ(expected_messages, messages);
}

// Failure case can't be tested as part of the rest of the suite
TEST_F(MultiExceptionHandlerTest, DISABLED_uncaught)
{
    MultiExceptionHandler catchme;
    CELER_TRY_ELSE(CELER_VALIDATE(false, << "derp"), catchme);
    // Program will terminate when catchme leaves scope
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
