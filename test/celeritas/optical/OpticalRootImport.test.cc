//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalRootImport.test.cc
//---------------------------------------------------------------------------//

#include "celeritas/LArSphereBase.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/MaterialParams.hh"
#include "celeritas/optical/ModelImporter.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

#define OpticalRootImportTest TEST_IF_CELERITAS_USE_ROOT(OpticalRootImportTest)
class OpticalRootImportTest : public RootTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "lar-sphere"sv;
    }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
};

TEST_F(OpticalRootImportTest, import_models)
{
    ModelImporter model_importer(
        this->imported_data(), this->optical_material(), this->material());

    EXPECT_TRUE(model_importer(ImportModelClass::absorption));
    EXPECT_TRUE(model_importer(ImportModelClass::rayleigh));
    // EXPECT_TRUE(model_importer(ImportModelClass::wls));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
