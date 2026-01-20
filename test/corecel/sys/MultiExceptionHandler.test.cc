//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/MultiExceptionHandler.hh"

#include <regex>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"

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
    explicit MockContextException(std::string msg = "some context")
        : msg_(std::move(msg))
    {
    }

    char const* what() const noexcept final { return msg_.c_str(); }

  private:
    std::string msg_;
};

//---------------------------------------------------------------------------//

class MultiExceptionHandlerTest : public ::celeritas::test::Test
{
  protected:
    MultiExceptionHandlerTest() : scoped_log_(&celeritas::self_logger()) {}

    ScopedLogStorer scoped_log_;
};

// Test that a single error throws as expected
TEST_F(MultiExceptionHandlerTest, single)
{
    MultiExceptionHandler capture_exception;
    EXPECT_TRUE(capture_exception.empty());
    CELER_TRY_HANDLE(CELER_RUNTIME_THROW("runtime", "first exception", ""),
                     capture_exception);
    EXPECT_FALSE(capture_exception.empty());

    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);
}

//! Test that four different messages all show up
TEST_F(MultiExceptionHandlerTest, multi)
{
    MultiExceptionHandler capture_exception;
    CELER_TRY_HANDLE(
        throw RuntimeError({"runtime", "first exception", "", "test.cc", 0}),
        capture_exception);
    for (auto i : range(3))
    {
        DebugErrorDetails deets{
            DebugErrorType::internal, "false", "test.cc", i};
        CELER_TRY_HANDLE(throw DebugError(std::move(deets)), capture_exception);
    }
    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);

    static char const* const expected_log_messages[] = {
        "[1/4]: runtime error: first exception\ntest.cc: failure",
        "[2/4]: test.cc:0:\nceleritas: internal assertion failed: false",
        "[3/4]: test.cc:1:\nceleritas: internal assertion failed: false",
        "[4/4]: test.cc:2:\nceleritas: internal assertion failed: false",
    };
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[]
        = {"critical", "critical", "critical", "critical"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

//! Test that similar messages are suppressed
TEST_F(MultiExceptionHandlerTest, multi_nested)
{
    MultiExceptionHandler capture_exception;
    CELER_TRY_HANDLE_CONTEXT(
        throw RuntimeError({"runtime", "it just got real", "", "test.cc", 1}),
        capture_exception,
        MockContextException{});
    for (auto i = 0; i < 4; ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(
            throw DebugError({DebugErrorType::internal, "false", "test.cc", 2}),
            capture_exception,
            MockContextException{"context " + std::to_string(i)});
    }

    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)),
                 MockContextException);

    static char const* const expected_log_messages[] = {
        R"([1/5]: runtime error: it just got real
test.cc:1: failure
    ...from some context)",
        R"([2/5]: test.cc:2:
celeritas: internal assertion failed: false
    ...from context 0)",
        "[3-5/5]: identical root cause to exception [2/5]",
    };
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages()) << scoped_log_;
    static char const* const expected_log_levels[]
        = {"critical", "critical", "critical"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

// Test that uncaught exceptions terminate the program
TEST_F(MultiExceptionHandlerTest, uncaught)
{
    EXPECT_DEATH(
        {
            MultiExceptionHandler catchme;
            CELER_TRY_HANDLE(CELER_VALIDATE(false, << "derp"), catchme);
            // Program will terminate when catchme leaves scope
        },
        "failed to clear exceptions from MultiExceptionHandler");
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
