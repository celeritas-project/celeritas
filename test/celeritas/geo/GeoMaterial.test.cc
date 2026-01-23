//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/GeoMaterial.test.cc
//---------------------------------------------------------------------------//
#include "corecel/data/CollectionStateStore.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoMaterialParams.hh"
#include "celeritas/geo/GeoMaterialView.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "celeritas_test.hh"

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
    std::string material_name(PhysMatId matid) const
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
        result.push_back(this->material_name(
            geo_mat_view.material_id(geo.impl_volume_id())));

        geo.find_next_step();
        geo.move_to_boundary();
        geo.cross_boundary();
    }
    return result;
}

//---------------------------------------------------------------------------//

#if CELERITAS_USE_ROOT
#    define CMS_TEST_BASE RootTestBase
#else
#    define CMS_TEST_BASE GeantTestBase
#endif

class SimpleCmsTest : public CMS_TEST_BASE, public GeoMaterialTestBase
{
  public:
    std::string_view gdml_basename() const override { return "simple-cms"sv; }
};

//---------------------------------------------------------------------------//

class Em3Test : public GeantTestBase, public GeoMaterialTestBase
{
    std::string_view gdml_basename() const override { return "testem3-flat"; }
};

//---------------------------------------------------------------------------//

class MultiLevelTest : public GeantTestBase, public GeoMaterialTestBase
{
    std::string_view gdml_basename() const final { return "multi-level"sv; }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(SimpleCmsTest, plus_z)
{
    auto materials = this->trace_materials({0, 0, 0}, {1, 0, 0});
    static char const* const expected_materials[]
        = {"vacuum", "Si", "Pb", "C", "Ti", "Fe", "vacuum"};
    EXPECT_VEC_EQ(expected_materials, materials);
}

TEST_F(Em3Test, plus_x)
{
    auto materials = this->trace_materials({19.01, 0, 0}, {1, 0, 0});
    static char const* const expected_materials[]
        = {"lAr", "Pb", "lAr", "vacuum"};
    EXPECT_VEC_EQ(expected_materials, materials);
}

TEST_F(MultiLevelTest, high)
{
    auto materials = this->trace_materials({-19.9, 7.5, 0}, {1, 0, 0});

    static char const* const expected_materials[] = {
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
    };
    EXPECT_VEC_EQ(expected_materials, materials);
}

TEST_F(MultiLevelTest, low)
{
    auto materials = this->trace_materials({-19.9, -7.5, 0}, {1, 0, 0});
    static char const* const expected_materials[] = {
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
        "Pb",
        "lAr",
    };
    EXPECT_VEC_EQ(expected_materials, materials);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
