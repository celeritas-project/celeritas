//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file JsonComparer.test.cc
//---------------------------------------------------------------------------//
#include "testdetail/JsonComparer.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace testdetail
{
namespace test
{
//---------------------------------------------------------------------------//

class JsonComparerTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(JsonComparerTest, parse_errors)
{
    JsonComparer compare;
    EXPECT_FALSE(compare("not valid json"));
    EXPECT_FALSE(compare("null", "blorp"));
}

TEST_F(JsonComparerTest, scalars)
{
    JsonComparer compare{real_type(0.001)};

    EXPECT_TRUE(compare("null", "null"));

    EXPECT_TRUE(compare("10"));
    EXPECT_FALSE(compare("10", "11"));

    EXPECT_TRUE(compare("10.0"));
    EXPECT_TRUE(compare("10.0", "10.0001"));
    EXPECT_FALSE(compare("10.0", "10.1"));
    //
    EXPECT_TRUE(compare("\"hi\"", "\"hi\""));
    EXPECT_FALSE(compare("\"hi\"", "\"bye\""));

    EXPECT_FALSE(compare("10.0", "10"));  // float to int
    EXPECT_FALSE(compare("10", "null"));  // float to null
}

TEST_F(JsonComparerTest, array)
{
    JsonComparer compare;

    EXPECT_TRUE(compare("[]", "[]"));
    EXPECT_TRUE(compare("[1, 2, 3]", "[1, 2, 3]"));
    EXPECT_FALSE(compare("[1, 2, 3]", "[2, 2, 3]"));
}

TEST_F(JsonComparerTest, object)
{
    JsonComparer compare{real_type(0.001)};

    EXPECT_TRUE(compare(R"json({"a": 1, "b": 2})json"));
    EXPECT_TRUE(
        compare(R"json({"a": 1, "b": 2})json", R"json({"b": 2, "a": 1})json"));
    EXPECT_FALSE(compare(R"json({"a": 1})json", R"json({"a": 1, "c": 2})json"));
    EXPECT_FALSE(
        compare(R"json({"a": 1, "b": 2})json", R"json({"a": 1, "c": 2})json"));
    EXPECT_FALSE(
        compare(R"json({"a": 1, "b": 2})json", R"json({"a": 2, "b": 1})json"));
}

TEST_F(JsonComparerTest, stringification)
{
    JsonComparer compare{real_type(0.001)};
    auto r = compare(R"json({"a": 1, "b": [1, 2, [0]]})json",
                     R"json({"a": 2, "b": [2, 3, [4, 5]]})json");
    EXPECT_STREQ(R"(JSON objects differ:
  value in .["a"]: expected 1, but got 2
  value in .["b"][0]: expected 1, but got 2
  value in .["b"][1]: expected 2, but got 3
  size in .["b"][2]: expected 1, but got 2)",
                 r.message());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace testdetail
}  // namespace celeritas
