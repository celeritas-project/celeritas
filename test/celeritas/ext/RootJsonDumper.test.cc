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

    std::string actual = [&imported] {
        std::ostringstream os;
        ScopedRootErrorHandler scoped_root_error;
        RootJsonDumper{os}(imported);
        scoped_root_error.throw_if_errors();
        return std::move(os).str();
    }();

    std::string const ref_path = this->test_data_path("celeritas", "")
                                 + "four-steel-slabs.root-dump.json";

    std::ifstream infile(ref_path);
    if (!infile)
    {
        ADD_FAILURE() << "failed to open reference file at '" << ref_path
                      << "': writing new reference file instead";
        infile.close();

        std::ofstream outfile(ref_path);
        CELER_VALIDATE(outfile,
                       << "failed to open reference file '" << ref_path
                       << "' for writing");
        outfile << actual << "\n";
        return;
    }

    std::stringstream expected;
    expected << infile.rdbuf();

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_EQ(expected.str(), actual)
            << "remove file at " << ref_path
            << " and rerun to generate new reference file";
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
