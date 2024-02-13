//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterial.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/CollectionStateStore.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "celeritas_test.hh"
#include "../RootTestBase.hh"
#include "../TestEm3Base.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeoMaterialTestBase : virtual public GlobalTestBase
{
  public:
    using VecString = std::vector<std::string>;

  protected:
    std::string material_name(MaterialId matid) const
    {
        if (!matid)
            return "---";
        return this->material()->id_to_label(matid).name;
    }

    VecString trace_materials(Real3 const& pos, Real3 dir);
};

auto GeoMaterialTestBase::trace_materials(Real3 const& pos_cm, Real3 dir)
    -> VecString
{
    CollectionStateStore<GeoStateData, MemSpace::host> host_state{
        this->geometry()->host_ref(), 1};
    // Geometry track view and mat view
    GeoTrackView geo(
        this->geometry()->host_ref(), host_state.ref(), TrackSlotId{0});
    GeoMaterialView geo_mat_view(this->geomaterial()->host_ref());

    // Track across layers to get a truly implementation-independent
    // comparison of material IDs encountered.
    VecString result;

    geo = {from_cm(pos_cm), make_unit_vector(dir)};
    while (!geo.is_outside())
    {
        result.push_back(
            this->material_name(geo_mat_view.material_id(geo.volume_id())));

        geo.find_next_step();
        geo.move_to_boundary();
        geo.cross_boundary();
    }
    return result;
}

//---------------------------------------------------------------------------//

#define SimpleCmsRoot TEST_IF_CELERITAS_USE_ROOT(SimpleCmsRoot)
class SimpleCmsRoot : public RootTestBase, public GeoMaterialTestBase
{
  public:
    std::string_view geometry_basename() const override
    {
        return "simple-cms"sv;
    }
    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }
};

//---------------------------------------------------------------------------//

#define TestEm3 TEST_IF_CELERITAS_GEANT(TestEm3)
class TestEm3 : public TestEm3Base, public GeoMaterialTestBase
{
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SimpleCmsRoot, plus_z)
{
    auto materials = this->trace_materials({0, 0, 0}, {1, 0, 0});
    static char const* const expected_materials[]
        = {"vacuum", "Si", "Pb", "C", "Ti", "Fe", "vacuum"};
    EXPECT_VEC_EQ(expected_materials, materials);
}

TEST_F(TestEm3, plus_x)
{
    auto materials = this->trace_materials({19.01, 0, 0}, {1, 0, 0});
    static char const* const expected_materials[]
        = {"lAr", "Pb", "lAr", "vacuum"};
    EXPECT_VEC_EQ(expected_materials, materials);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
