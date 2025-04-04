//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Test.cc
//---------------------------------------------------------------------------//
#include "Test.hh"

#include <algorithm>
#include <cctype>
#include <fstream>

#include "celeritas_test_config.h"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/sys/Environment.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Get the path to a test file at `{source}/test/{subdir}/data/{filename}`.
 *
 * \post The given input file exists. (ifstream is used to check this)
 */
std::string
Test::test_data_path(std::string_view subdir, std::string_view filename)
{
    std::ostringstream os;
    os << celeritas_source_dir << "/test/" << subdir << "/data/" << filename;

    std::string path = os.str();
    if (!filename.empty())
    {
        CELER_VALIDATE(std::ifstream(path).good(),
                       << "Failed to open test data file at '" << path << "'");
    }
    return path;
}

//---------------------------------------------------------------------------//
/*!
 * Generate test-unique filename.
 */
std::string Test::make_unique_filename(std::string_view ext)
{
    // Get filename based on unit test name
    ::testing::TestInfo const* const test_info
        = ::testing::UnitTest::GetInstance()->current_test_info();
    CELER_ASSERT(test_info);

    // Convert test case to lowercase
    std::string case_name = test_info->test_case_name();
    std::string test_name = test_info->name();

    // Delete "Test", "DISABLED"; make lowercase; make sure not empty
    for (auto* str : {&case_name, &test_name})
    {
        for (std::string replace : {"Test", "DISABLED"})
        {
            auto pos = str->rfind(replace);
            while (pos != std::string::npos)
            {
                str->replace(pos, replace.size(), "", 0);
                pos = str->rfind(replace);
            }
        }
        std::transform(
            str->begin(), str->end(), str->begin(), [](unsigned char c) {
                return std::tolower(c);
            });
        if (str->empty())
        {
            *str = "test";
        }

        {
            // Strip leading underscores
            auto iter = str->begin();
            while (iter != str->end() && *iter == '_')
                ++iter;
            str->erase(str->begin(), iter);
        }
        {
            // Strip trailing underscores
            auto iter = str->rbegin();
            while (iter != str->rend() && *iter == '_')
                ++iter;
            str->erase(iter.base(), str->end());
        }

        // Replace slashes/underscores with dashes
        for (char& c : *str)
        {
            if (c == '/' || c == '_')
                c = '-';
        }
    }

    // Concatenate string
    std::ostringstream os;
    os << case_name << '-' << test_name;

    if (filename_counter_)
    {
        os << '-' << filename_counter_;
    }
    ++filename_counter_;

    os << ext;

    return os.str();
}

//---------------------------------------------------------------------------//
/*!
 * True if strict testing is required.
 *
 * This is set during CI tests so that "loose" tests (which might depend on the
 * environment) are enabled inside the CI.
 */
bool Test::strict_testing()
{
    std::string const& envstr = ::celeritas::getenv("CELER_TEST_STRICT");
    if (envstr == "0")
    {
        return false;
    }
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_GEANT4)
    {
        // Disable strict testing for Geant4
        return false;
    }
    if (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE)
    {
        // Disable strict testing for single precision
        return false;
    }
    if (CELERITAS_UNITS != CELERITAS_UNITS_CGS)
    {
        // Disable strict testing for non-CLHEP units
        return false;
    }
    return !envstr.empty();
}

//---------------------------------------------------------------------------//
// Provide a definition for the "inf" value. (This is needed by C++ < 17 so
// that the adddress off the static value can be taken.)
constexpr double Test::inf;
constexpr real_type Test::coarse_eps;
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
