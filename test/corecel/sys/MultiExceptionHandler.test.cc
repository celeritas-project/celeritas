//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/MultiExceptionHandler.hh"

#include <regex>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/MpiCommunicator.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Helper classes
class MockContextException : public std::exception
{
  public:
    char const* what() const noexcept final { return "some context"; }
};

//---------------------------------------------------------------------------//

class MultiExceptionHandlerTest : public ::celeritas::test::Test
{
  protected:
    MultiExceptionHandlerTest() : store_log_(&celeritas::self_logger()) {}

    ScopedLogStorer store_log_;
};

TEST_F(MultiExceptionHandlerTest, single)
{
    MultiExceptionHandler capture_exception;
    EXPECT_TRUE(capture_exception.empty());
    CELER_TRY_HANDLE(
        throw RuntimeError::from_validate("first exception", "", "here", 1),
        capture_exception);
    EXPECT_FALSE(capture_exception.empty());

    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);
}

TEST_F(MultiExceptionHandlerTest, multi)
{
    MultiExceptionHandler capture_exception;
    CELER_TRY_HANDLE(
        throw RuntimeError::from_validate("first exception", "", "here", 1),
        capture_exception);
    for (auto i : range(3))
    {
        DebugErrorDetails deets{
            DebugErrorType::internal, "false", "test.cc", i};
        CELER_TRY_HANDLE(throw DebugError(deets), capture_exception);
    }
    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);

    static char const* const expected_messages[]
        = {"ignoring exception: test.cc:0:\nceleritas: internal assertion "
           "failed: false",
           "ignoring exception: test.cc:1:\nceleritas: internal assertion "
           "failed: false",
           "ignoring exception: test.cc:2:\nceleritas: internal assertion "
           "failed: false"};
    EXPECT_VEC_EQ(expected_messages, store_log_.messages());

    static char const* const expected_log_levels[]
        = {"critical", "critical", "critical"};
    EXPECT_VEC_EQ(expected_log_levels, store_log_.levels());
}

TEST_F(MultiExceptionHandlerTest, multi_nested)
{
    MultiExceptionHandler capture_exception;
    CELER_TRY_HANDLE_CONTEXT(
        throw RuntimeError::from_validate("first exception", "", "here", 1),
        capture_exception,
        MockContextException{});
    DebugErrorDetails deets{DebugErrorType::internal, "false", "test.cc", 2};
    CELER_TRY_HANDLE_CONTEXT(
        throw DebugError(deets), capture_exception, MockContextException{});
    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)),
                 MockContextException);

    static char const* const expected_messages[]
        = {"ignoring exception: test.cc:2:\nceleritas: internal assertion "
           "failed: false\n... from: some context"};
    EXPECT_VEC_EQ(expected_messages, store_log_.messages());
}

// Failure case can't be tested as part of the rest of the suite
TEST_F(MultiExceptionHandlerTest, DISABLED_uncaught)
{
    MultiExceptionHandler catchme;
    CELER_TRY_HANDLE(CELER_VALIDATE(false, << "derp"), catchme);
    // Program will terminate when catchme leaves scope
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
