//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/RootJsonDumper.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/RootJsonDumper.hh"

#include <sstream>

#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/ScopedRootErrorHandler.hh"
#include "celeritas/io/ImportData.hh"

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
    ImportData imported;
    {
        ScopedRootErrorHandler scoped_root_error;
        RootImporter import(
            this->test_data_path("celeritas", "four-steel-slabs.root"));
        imported = import();
        scoped_root_error.throw_if_errors();
    }

    std::ostringstream os;
    {
        ScopedRootErrorHandler scoped_root_error;
        RootJsonDumper{&os}(imported);
        scoped_root_error.throw_if_errors();
    }
    EXPECT_JSON_EQ("", os.str());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
