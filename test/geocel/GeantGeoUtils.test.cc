//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoUtils.test.cc
//---------------------------------------------------------------------------//
#include "geocel/GeantGeoUtils.hh"

#include <sstream>
#include <vector>
#include <G4LogicalVolume.hh>

#include "geocel/Types.hh"
#include "geocel/g4/GeantGeoTrackView.hh"

#include "UnitUtils.hh"
#include "celeritas_test.hh"
#include "g4/GeantGeoTestBase.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
// Get volume names for a bunch of G4LV*
template<class InputIterator>
decltype(auto) get_vol_names(InputIterator iter, InputIterator stop)
{
    std::vector<std::string> result;
    for (; iter != stop; ++iter)
    {
        CELER_ASSERT(*iter);
        result.push_back((*iter)->GetName());
    }
    std::sort(result.begin(), result.end());
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

class GeantGeoUtilsTest : public GeantGeoTestBase
{
  public:
    void SetUp() override
    {
        // Build geometry during setup
        ASSERT_TRUE(this->geometry());
    }
};

//---------------------------------------------------------------------------//
class SolidsTest : public GeantGeoUtilsTest
{
    std::string_view gdml_basename() const override { return "solids"; }
};

TEST_F(SolidsTest, find_geant_volumes)
{
    auto vols = find_geant_volumes({"box500", "trd3", "trd1"});
    auto vol_names = get_vol_names(vols.begin(), vols.end());
    static char const* const expected_vol_names[] = {"box500", "trd1", "trd3"};
    EXPECT_VEC_EQ(expected_vol_names, vol_names);
}

TEST_F(SolidsTest, find_geant_volumes_missing)
{
    EXPECT_THROW(find_geant_volumes({"box500", "trd3", "turd3"}), RuntimeError);
}

TEST_F(SolidsTest, find_geant_volumes_duplicate)
{
    auto vols = find_geant_volumes({"trd3_refl"});
    auto vol_names = get_vol_names(vols.begin(), vols.end());
    static char const* const expected_vol_names[] = {"trd3_refl", "trd3_refl"};
    EXPECT_VEC_EQ(expected_vol_names, vol_names);
}

//---------------------------------------------------------------------------//
class MultiLevelTest : public GeantGeoUtilsTest
{
    std::string_view gdml_basename() const override { return "multi-level"; }
};

TEST_F(MultiLevelTest, printable_nav)
{
    auto get_nav_str = [this](Real3 const& pos) {
        auto geo = this->make_geo_track_view();
        geo = this->make_initializer(pos, {1, 0, 0});
        std::ostringstream os;
        os << StreamableNavHistory{geo.track_view().nav_history()};
        return std::move(os).str();
    };

    EXPECT_EQ(R"({{pv='boxtri', lv=27='tri'} -> {pv='topbox1', lv=28='box'}})",
              get_nav_str({12.5, 7.5, 0}));
    EXPECT_EQ(
        R"({{pv='boxtri', lv=32='tri_refl'} -> {pv='topbox4', lv=30='box_refl'}})",
        get_nav_str({12.5, -7.5, 0}));
    EXPECT_EQ(R"({{pv='boxtri', lv=27='tri'} -> {pv='topbox2', lv=28='box'}})",
              get_nav_str({-7.5, 7.5, 0}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
