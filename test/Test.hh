//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <string>
#include <gtest/gtest.h>

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Googletest test harness for Celeritas codes.
 *
 * The test harness is constructed and destroyed once per subtest.
 */
class Test : public ::testing::Test
{
  public:
    Test() = default;

    // Generate test-unique filename
    std::string make_unique_filename(const char* ext = "");

    // Get the path to a test file in `{source}/test/{subdir}/data/{filename}`
    static std::string test_data_path(const char* subdir, const char* filename);

    // True if CELER_TEST_STRICT is set (under CI)
    static bool strict_testing();

    // Define "inf" value for subclass testing
    static constexpr double inf = HUGE_VAL;

  private:
    int filename_counter_ = 0;
};

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
