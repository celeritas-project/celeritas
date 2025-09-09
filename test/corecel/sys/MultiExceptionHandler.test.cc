//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/MultiExceptionHandler.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/MultiExceptionHandler.hh"

#include <regex>

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
    char const* what() const noexcept final { return "some context"; }
};

//---------------------------------------------------------------------------//

class MultiExceptionHandlerTest : public ::celeritas::test::Test
{
  protected:
    MultiExceptionHandlerTest() : scoped_log_(&celeritas::self_logger()) {}

    ScopedLogStorer scoped_log_;
};

TEST_F(MultiExceptionHandlerTest, single)
{
    MultiExceptionHandler capture_exception;
    EXPECT_TRUE(capture_exception.empty());
    CELER_TRY_HANDLE(CELER_RUNTIME_THROW("runtime", "first exception", ""),
                     capture_exception);
    EXPECT_FALSE(capture_exception.empty());

    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);
}

TEST_F(MultiExceptionHandlerTest, multi)
{
    MultiExceptionHandler capture_exception;
    CELER_TRY_HANDLE(CELER_RUNTIME_THROW("runtime", "first exception", ""),
                     capture_exception);
    for (auto i : range(3))
    {
        DebugErrorDetails deets{
            DebugErrorType::internal, "false", "test.cc", i};
        CELER_TRY_HANDLE(throw DebugError(std::move(deets)), capture_exception);
    }
    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)), RuntimeError);

    static char const* const expected_log_messages[] = {
        R"(Suppressed exception from parallel thread: test.cc:0:
celeritas: internal assertion failed: false)",
        R"(Suppressed exception from parallel thread: test.cc:1:
celeritas: internal assertion failed: false)",
        R"(Suppressed exception from parallel thread: test.cc:2:
celeritas: internal assertion failed: false)",
    };
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[]
        = {"critical", "critical", "critical"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(MultiExceptionHandlerTest, multi_nested)
{
    MultiExceptionHandler capture_exception;
    CELER_TRY_HANDLE_CONTEXT(
        CELER_RUNTIME_THROW("runtime", "first exception", ""),
        capture_exception,
        MockContextException{});
    DebugErrorDetails deets{DebugErrorType::internal, "false", "test.cc", 2};
    for (auto i = 0; i < 4; ++i)
    {
        CELER_TRY_HANDLE_CONTEXT(throw DebugError(std::move(deets)),
                                 capture_exception,
                                 MockContextException{});
    }

    EXPECT_THROW(log_and_rethrow(std::move(capture_exception)),
                 MockContextException);

    static char const* const expected_log_messages[] = {
        R"(Suppressed exception from parallel thread: test.cc:2:
celeritas: internal assertion failed: false
... from some context)",
        "Suppressed 3 similar exceptions",
    };
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"critical", "warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
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
