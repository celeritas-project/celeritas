//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <string>
#include <string_view>
#include <gtest/gtest.h>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * \namespace test
 *
 * This sub-namespace should enclose all Celeritas unit tests and test harness
 * code. By being a sub-namspace, no \c using declarations are needed to access
 * Celeritas functionality, and no symbol conflicts with main Celeritas code
 * will accidentally occur.
 */
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
    std::string make_unique_filename(std::string_view ext = {});

    // Get the path to a test file in `{source}/test/{subdir}/data/{filename}`
    static std::string
    test_data_path(std::string_view subdir, std::string_view filename);

    // True if CELER_TEST_STRICT is set (under CI)
    static bool strict_testing();

    // Define "inf" value for subclass testing
    static constexpr double inf = HUGE_VAL;
    static constexpr float inff = HUGE_VALF;

  private:
    int filename_counter_ = 0;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
