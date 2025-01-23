//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StringUtils.test.cc
//---------------------------------------------------------------------------//
#include "corecel/io/StringUtils.hh"

#include <string_view>

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(StringUtils, starts_with)
{
    EXPECT_TRUE(starts_with("prefix", "pre"));
    EXPECT_FALSE(starts_with("abcd", "b"));
    EXPECT_FALSE(starts_with("a", "abcd"));
    EXPECT_TRUE(starts_with("", ""));
}

//---------------------------------------------------------------------------//
TEST(StringUtils, ends_with)
{
    EXPECT_TRUE(ends_with("prefix", "fix"));
    EXPECT_FALSE(ends_with("abcd", "c"));
    EXPECT_FALSE(ends_with("d", "abcd"));
    EXPECT_TRUE(ends_with("", ""));
}

//---------------------------------------------------------------------------//
TEST(StringUtils, is_ignored_trailing)
{
    EXPECT_TRUE(is_ignored_trailing(' '));
    EXPECT_TRUE(is_ignored_trailing('\a'));
    EXPECT_TRUE(is_ignored_trailing('\n'));
    EXPECT_FALSE(is_ignored_trailing('a'));
    EXPECT_FALSE(is_ignored_trailing('!'));
}

//---------------------------------------------------------------------------//
TEST(StringUtils, trim)
{
    using namespace std::literals::string_view_literals;

    EXPECT_EQ(""sv, trim(""));
    EXPECT_EQ(""sv, trim(" "));
    EXPECT_EQ("what ho"sv, trim(" what ho  "));
    EXPECT_EQ("what ho"sv, trim("\twhat ho \a \n"));
}

//---------------------------------------------------------------------------//
TEST(StringUtils, tolower)
{
    EXPECT_EQ("", tolower(""));
    EXPECT_EQ(" hello!  ", tolower(" HeLLo!  "));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
