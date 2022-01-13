//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Environment.test.cc
//---------------------------------------------------------------------------//
#include "comm/Environment.hh"

#include <vector>

#include "celeritas_config.h"
#if CELERITAS_USE_JSON
#    include "comm/EnvironmentIO.json.hh"
#endif

#include "celeritas_test.hh"

using celeritas::Environment;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class EnvironmentTest : public celeritas::Test
{
  protected:
    void SetUp() override {}
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(EnvironmentTest, local)
{
    Environment env;
    EXPECT_TRUE(env.begin() == env.end());
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
  ENVTEST_UNSET: '',
  ENVTEST_EMPTY: '',
  ENVTEST_ZERO: '0',
  ENVTEST_ONE: '1',
})",
              os.str());
}

TEST_F(EnvironmentTest, global)
{
    Environment& env = celeritas::environment();

    // Function to return whether a environment variable has been accessed
    auto found_env = [&env](const std::string& s) -> bool {
        return std::find_if(env.begin(),
                            env.end(),
                            [&s](const Environment::value_type& kv) {
                                return s == kv.first;
                            })
               != env.end();
    };

    EXPECT_FALSE(found_env("ENVTEST_ONE"));
    EXPECT_EQ("1", celeritas::getenv("ENVTEST_ONE"));
    EXPECT_TRUE(found_env("ENVTEST_ONE"));
}

TEST_F(EnvironmentTest, json)
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
        nlohmann::json    out{env};
        static const char expected[]
            = R"json([{"ENVTEST_CUSTOM":"custom","ENVTEST_ONE":"111111","ENVTEST_ZERO":"0"}])json";
        EXPECT_EQ(std::string(expected), std::string(out.dump()));
    }
#else
    GTEST_SKIP() << "JSON is disabled";
#endif
}
