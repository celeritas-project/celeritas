//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootJsonDumper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/RootJsonDumper.hh"

#include <fstream>
#include <sstream>

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportDataTrimmer.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class RootJsonDumperTest : public ::celeritas::test::Test
{
};

TEST_F(RootJsonDumperTest, all)
{
    // Import data
    ImportData imported = [this] {
        ScopedRootErrorHandler scoped_root_error;
        RootImporter import(
            this->test_data_path("celeritas", "four-steel-slabs.root"));
        auto result = import();
        scoped_root_error.throw_if_errors();
        return result;
    }();

    // Trim data
    {
        ImportDataTrimmer::Input inp;
        inp.materials = true;
        inp.physics = true;
        inp.mupp = true;
        inp.max_size = 2;
        ImportDataTrimmer trim{inp};
        trim(imported);
    }

    std::string str = [&imported] {
        std::ostringstream os;
        ScopedRootErrorHandler scoped_root_error;
        RootJsonDumper{os}(imported);
        scoped_root_error.throw_if_errors();
        return std::move(os).str();
    }();

    std::string expected = [this] {
        std::ifstream infile(this->test_data_path(
            "celeritas", "four-steel-slabs.root-dump.json"));
        std::ostringstream os;
        std::string line;
        while (std::getline(infile, line))
        {
            os << line << "\n";
        }
        return std::move(os).str();
    }();

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(expected, str);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
