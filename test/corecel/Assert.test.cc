//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/Assert.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"

#include <string>
#include <vector>

#include "corecel/io/Repr.hh"
#include "corecel/sys/Environment.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class AssertTest : public ::celeritas::test::Test
{
  protected:
    static void SetUpTestSuite()
    {
        EXPECT_FALSE(celeritas::getenv("CELER_COLOR").empty()
                     && celeritas::getenv("GTEST_COLOR").empty())
            << "Color must be enabled for this test";
        EXPECT_FALSE(celeritas::getenv("CELER_LOG").empty())
            << "Logging (for verbose output) must be enabled for this test";
    }
};

TEST_F(AssertTest, debug_error)
{
    DebugErrorDetails details;
    details.which = DebugErrorType::internal;
    details.condition = "2 + 2 == 5";
    details.file = "Assert.test.cc";
    details.line = 123;

    EXPECT_STREQ(
        "\x1B[37;1mAssert.test.cc:123:\x1B[0m\nceleritas: \x1B[31;1minternal "
        "assertion failed: \x1B[37;2m2 + 2 == 5\x1B[0m",
        DebugError{std::move(details)}.what());
}

TEST_F(AssertTest, runtime_error)
{
    try
    {
        CELER_NOT_CONFIGURED("foo");
    }
    catch (RuntimeError const& e)
    {
        EXPECT_TRUE(std::string{e.what()}.find("configuration error:")
                    != std::string::npos);
        EXPECT_TRUE(std::string{e.what()}.find("required dependency is "
                                               "disabled in this build: foo")
                    != std::string::npos)
            << e.what();
    }
    try
    {
        CELER_NOT_IMPLEMENTED("bar");
    }
    catch (RuntimeError const& e)
    {
        EXPECT_TRUE(std::string{e.what()}.find("implementation error:")
                    != std::string::npos);
        EXPECT_TRUE(std::string{e.what()}.find("feature is not yet "
                                               "implemented: bar")
                    != std::string::npos)
            << e.what();
    }
    try
    {
        CELER_VALIDATE(false, << "this is not OK");
    }
    catch (RuntimeError const& e)
    {
        EXPECT_TRUE(std::string{e.what()}.find("runtime error: \x1B[0mthis is "
                                               "not OK")
                    != std::string::npos)
            << e.what();
    }
}

TEST_F(AssertTest, runtime_error_variations)
{
    std::vector<std::string> messages;
    // Loop over all combinations of missing data
    for (auto bitmask : range(1 << 4))
    {
        RuntimeErrorDetails details;
        if (bitmask & 0x1)
        {
            details.which = "runtime";
        }
        if (bitmask & 0x2)
        {
            details.what = "bad things happened";
        }
        if (bitmask & 0x4)
        {
            details.condition = "2 + 2 == 5";
        }
        if (bitmask & 0x8)
        {
            details.file = "Assert.test.cc";
            if (bitmask & 0x1)
            {
                details.line = 123;
            }
        }
        // Generate the message
        RuntimeError err{std::move(details)};
        messages.push_back(err.what());
    }

    // clang-format off
    static char const* const expected_messages[] = {
        "\x1b[31;1munknown error: \x1b[0m\n\x1b[37;2munknown source:\x1b[0m failure",
        "\x1b[31;1mruntime error: \x1b[0m\n\x1b[37;2munknown source:\x1b[0m failure",
        "\x1b[31;1munknown error: \x1b[0mbad things happened\n\x1b[37;2munknown source:\x1b[0m failure",
        "\x1b[31;1mruntime error: \x1b[0mbad things happened\n\x1b[37;2munknown source:\x1b[0m failure",
        "\x1b[31;1munknown error: \x1b[0m\n\x1b[37;1munknown source:\x1b[0m '2 + 2 == 5' failed",
        "\x1b[31;1mruntime error: \x1b[0m\n\x1b[37;1munknown source:\x1b[0m '2 + 2 == 5' failed",
        "\x1b[31;1munknown error: \x1b[0mbad things happened\n\x1b[37;1munknown source:\x1b[0m '2 + 2 == 5' failed",
        "\x1b[31;1mruntime error: \x1b[0mbad things happened\n\x1b[37;1munknown source:\x1b[0m '2 + 2 == 5' failed",
        "\x1b[31;1munknown error: \x1b[0m\n\x1b[37;2mAssert.test.cc:\x1b[0m failure",
        "\x1b[31;1mruntime error: \x1b[0m\n\x1b[37;2mAssert.test.cc:123:\x1b[0m failure",
        "\x1b[31;1munknown error: \x1b[0mbad things happened\n\x1b[37;2mAssert.test.cc:\x1b[0m failure",
        "\x1b[31;1mruntime error: \x1b[0mbad things happened\n\x1b[37;2mAssert.test.cc:123:\x1b[0m failure",
        "\x1b[31;1munknown error: \x1b[0m\n\x1b[37;1mAssert.test.cc:\x1b[0m '2 + 2 == 5' failed",
        "\x1b[31;1mruntime error: \x1b[0m\n\x1b[37;1mAssert.test.cc:123:\x1b[0m '2 + 2 == 5' failed",
        "\x1b[31;1munknown error: \x1b[0mbad things happened\n\x1b[37;1mAssert.test.cc:\x1b[0m '2 + 2 == 5' failed",
        "\x1b[31;1mruntime error: \x1b[0mbad things happened\n\x1b[37;1mAssert.test.cc:123:\x1b[0m '2 + 2 == 5' failed",
    };
    // clang-format on

    EXPECT_VEC_EQ(expected_messages, messages);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
