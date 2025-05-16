//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>
#include <string>
#include <string_view>
#include <gtest/gtest.h>  // IWYU pragma: export

#include "corecel/Config.hh"

namespace celeritas
{

/*!
 * This sub-namespace should enclose all Celeritas unit tests and test harness
 * code.
 *
 * By being a sub-namspace, no \c using declarations are needed to access
 * Celeritas functionality, and no symbol conflicts with main Celeritas code
 * will accidentally occur.
 */
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Googletest test harness for Celeritas codes.
 *
 * The test harness is constructed and destroyed once per subtest. It contains
 * helper functions and data commonly needed in Celeritas tests.
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

    // Define coarse epsilon (sqrt typical precision)

#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
    static constexpr double fine_eps = 1e-12;
#elif CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
    static constexpr float fine_eps = 1e-6f;
#endif

#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE
    static constexpr double coarse_eps = 1e-6;
#elif CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
    static constexpr float coarse_eps = 1e-3f;
#endif

  private:
    int filename_counter_ = 0;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
