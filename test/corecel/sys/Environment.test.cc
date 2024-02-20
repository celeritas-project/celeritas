//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/Environment.test.cc
//---------------------------------------------------------------------------//
#include "corecel/sys/Environment.hh"

#include <vector>

#include "celeritas_config.h"
#if CELERITAS_USE_JSON
#    include "corecel/sys/EnvironmentIO.json.hh"
#endif

#include "celeritas_test.hh"

namespace celeritas
{
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
    env.insert({"ENVTEST_ZERO", "2"});
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
    EXPECT_EQ("1", environment()["ENVTEST_ONE"]);
    EXPECT_EQ("1", getenv("ENVTEST_ONE"));
}

TEST(EnvironmentTest, merge)
{
    Environment sys;
    sys.insert({"FOO", "foo"});
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

TEST(EnvironmentTest, TEST_IF_CELERITAS_JSON(json))
{
#if CELERITAS_USE_JSON
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
        nlohmann::json out{env};
        EXPECT_JSON_EQ(
            R"json([{"ENVTEST_CUSTOM":"custom","ENVTEST_ONE":"111111","ENVTEST_ZERO":"000000"}])json",
            out.dump());
    }
#endif
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
