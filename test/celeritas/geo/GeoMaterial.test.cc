//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterial.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/CollectionStateStore.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoMaterialTest : public RootTestBase
{
  public:
    char const* geometry_basename() const override { return "simple-cms"; }

    SPConstTrackInit build_init() final { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() final { CELER_ASSERT_UNREACHABLE(); }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(GeoMaterialTest, host)
{
    // Geometry track view and mat view
    auto const& geo_params = *this->geometry();
    auto const& mat_params = *this->material();
    CollectionStateStore<GeoStateData, MemSpace::host> geo_state(
        geo_params.host_ref(), 1);
    GeoTrackView geo(geo_params.host_ref(), geo_state.ref(), ThreadId{0});
    GeoMaterialView geo_mat_view(this->geomaterial()->host_ref());

    // Track across layers to get a truly implementation-independent
    // comparison of material IDs encountered.
    std::vector<std::string> materials;

    geo = {{0, 0, 0}, {1, 0, 0}};
    while (!geo.is_outside())
    {
        MaterialId matid = geo_mat_view.material_id(geo.volume_id());

        materials.push_back(matid ? mat_params.id_to_label(matid).name
                                  : "[invalid]");

        geo.find_next_step();
        geo.move_to_boundary();
        geo.cross_boundary();
    }

    // PRINT_EXPECTED(materials);
    static const std::string expected_materials[]
        = {"vacuum", "Si", "Pb", "C", "Ti", "Fe", "vacuum"};
    EXPECT_VEC_EQ(expected_materials, materials);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
