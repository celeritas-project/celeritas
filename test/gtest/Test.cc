//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Test.cc
//---------------------------------------------------------------------------//
#include "Test.hh"

#include <algorithm>
#include <cctype>
#include <fstream>
#include "base/Assert.hh"
#include "detail/TestConfig.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get the path to a test file at `{source}/test/{subdir}/data/{filename}`.
 *
 * \post The given input file exists. (ifstream is used to check this)
 */
std::string Test::test_data_path(const char* subdir, const char* filename)
{
    std::ostringstream os;
    os << detail::source_dir << "/test/" << subdir << "/data/" << filename;

    std::string result = os.str();
    CELER_VALIDATE(std::ifstream(result).good(),
                   << "Failed to open test data file at " << result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Generate test-unique filename.
 */
std::string Test::make_unique_filename(const char* ext)
{
    CELER_EXPECT(ext);

    // Get filename based on unit test name
    const ::testing::TestInfo* const test_info
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
// Provide a definition for the "inf" value. (This is needed by C++ < 17 so
// that the adddress off the static value can be taken.)
constexpr double Test::inf;

//---------------------------------------------------------------------------//
} // namespace celeritas
