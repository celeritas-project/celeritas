//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <gtest/gtest.h>
#include <cmath>

namespace celeritas
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

    // Get the path to a test file in `{source}/test/{subdir}/data/{filename}`
    static std::string test_data_path(const char* subdir, const char* filename);

    // Define "inf" value for subclass testing
    static constexpr double inf = HUGE_VAL;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
