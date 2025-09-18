//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Environment.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/Environment.hh"

#include <vector>

#include "corecel/Config.hh"

#include "corecel/sys/EnvironmentIO.json.hh"

#include "celeritas_test.hh"

namespace celeritas
{
inline bool operator==(GetenvFlagResult a, GetenvFlagResult b)
{
    return a.value == b.value && a.defaulted == b.defaulted;
}
namespace test
{
//---------------------------------------------------------------------------//
// WARNING: this test will only pass through CTest because our CMake script
// sets a bunch of environment variables.
TEST(EnvironmentTest, local)
{
    Environment env;
    EXPECT_EQ("1", env["ENVTEST_ONE"]);
    EXPECT_EQ("0", env["ENVTEST_ZERO"]);
    EXPECT_EQ("", env["ENVTEST_EMPTY"]);
    EXPECT_EQ("", env["ENVTEST_UNSET"]);

    // Insert shouldn't override existing value
    EXPECT_FALSE(env.insert({"ENVTEST_ZERO", "2"}));
    EXPECT_EQ("0", env["ENVTEST_ZERO"]);

    std::ostringstream os;
    os << env;
    EXPECT_EQ(R"({
  ENVTEST_ONE: '1',
  ENVTEST_ZERO: '0',
  ENVTEST_EMPTY: '',
  ENVTEST_UNSET: '',
})",
              os.str());
}

TEST(EnvironmentTest, global)
{
    environment() = {};
    EXPECT_EQ("", getenv("ENVTEST_EMPTY"));

    EXPECT_EQ((GetenvFlagResult{false, false}),
              getenv_flag("ENVTEST_ZERO", false));
    EXPECT_EQ((GetenvFlagResult{true, false}),
              getenv_flag("ENVTEST_ONE", false));
    EXPECT_EQ((GetenvFlagResult{false, true}),
              getenv_flag("ENVTEST_EMPTY", false));
    EXPECT_EQ((GetenvFlagResult{true, true}),
              getenv_flag("ENVTEST_EMPTY", true));
    EXPECT_EQ((GetenvFlagResult{true, true}),
              getenv_flag("ENVTEST_NEW_T", true));
    EXPECT_EQ((GetenvFlagResult{false, true}),
              getenv_flag("ENVTEST_NEW_F", false));

    EXPECT_EQ("1", environment()["ENVTEST_ONE"]);
    EXPECT_EQ("0", getenv("ENVTEST_ZERO"));
    EXPECT_EQ("1", getenv("ENVTEST_ONE"));
    EXPECT_EQ("", getenv("ENVTEST_EMPTY"));

    environment().insert({"ENVTEST_FALSE", "false"});
    environment().insert({"ENVTEST_TRUE", "true"});
    EXPECT_EQ((GetenvFlagResult{false, false}),
              getenv_flag("ENVTEST_FALSE", false));
    EXPECT_EQ((GetenvFlagResult{true, false}),
              getenv_flag("ENVTEST_TRUE", false));
}

TEST(EnvironmentTest, global_overrides)
{
    auto& env = environment();

    // Reset already-read variables
    env = {};
    // Override a system environment variable
    EXPECT_TRUE(env.insert({"ENVTEST_ONE", "f"}));
    // Check that getenv_flag works
    EXPECT_EQ((GetenvFlagResult{false, false}),
              getenv_flag("ENVTEST_ONE", true));

    // This should pull from the system environment and store the saved result
    EXPECT_EQ((GetenvFlagResult{false, false}),
              getenv_flag("ENVTEST_ZERO", true));
    EXPECT_TRUE(env.find("ENVTEST_ZERO") != env.cend());
}

TEST(EnvironmentTest, merge)
{
    Environment sys;
    EXPECT_TRUE(sys.insert({"FOO", "foo"}));
    sys.insert({"BAR", "bar"});
    Environment input;
    input.insert({"FOO", "foo2"});
    input.insert({"BAZ", "baz"});
    sys.merge(input);

    std::ostringstream os;
    os << sys;
    EXPECT_EQ(R"({
  FOO: 'foo',
  BAR: 'bar',
  BAZ: 'baz',
})",
              os.str());
}

TEST(EnvironmentTest, json)
{
    // Pre-set one environment variable
    Environment env;
    EXPECT_EQ("0", env["ENVTEST_ZERO"]);

    {
        // Update environment
        nlohmann::json myenv{{"ENVTEST_ZERO", "000000"},
                             {"ENVTEST_ONE", "111111"},
                             {"ENVTEST_CUSTOM", "custom"}};
        myenv.get_to(env);
    }
    {
        // Save environment
        nlohmann::json out = env;
        EXPECT_JSON_EQ(
            R"json({"ENVTEST_CUSTOM":"custom","ENVTEST_ONE":"111111","ENVTEST_ZERO":"000000"})json",
            out.dump());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
