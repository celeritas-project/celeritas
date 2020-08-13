//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Test.cc
//---------------------------------------------------------------------------//
#include "Test.hh"

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
std::string Test::test_data_path(const char* subdir, const char* filename) const
{
    std::ostringstream os;
    os << detail::source_dir << "/test/" << subdir << "/data/" << filename;

    std::string result = os.str();
    ENSURE(std::ifstream(result).good());
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
