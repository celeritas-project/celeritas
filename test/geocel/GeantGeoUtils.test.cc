//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoUtils.test.cc
//---------------------------------------------------------------------------//
#include "geocel/GeantGeoUtils.hh"

#include <algorithm>
#include <G4LogicalVolume.hh>
#include <G4Navigator.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHistory.hh>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"

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
    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }

    void SetUp() override
    {
        // Build geometry during setup
        ASSERT_TRUE(this->geometry());
    }
};

//---------------------------------------------------------------------------//
class SolidsTest : public GeantGeoUtilsTest
{
    std::string geometry_basename() const override { return "solids"; }
};

TEST_F(SolidsTest, write_geant_geometry)
{
    auto* world = this->geometry()->world();
    ASSERT_TRUE(world);

    ScopedLogStorer scoped_log_{&celeritas::world_logger(), LogLevel::warning};
    write_geant_geometry(world, this->make_unique_filename(".gdml"));

    static char const* const expected_log_messages[] = {
        R"(Geant4 regions have not been set up: skipping export of energy cuts and regions)"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

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
class FourLevelsTest : public GeantGeoUtilsTest
{
    std::string geometry_basename() const override { return "four-levels"; }
};

TEST_F(FourLevelsTest, printable_nav)
{
    G4Navigator navi;
    G4TouchableHistory touchable;
    navi.SetWorldVolume(
        const_cast<G4VPhysicalVolume*>(this->geometry()->world()));
    navi.LocateGlobalPointAndUpdateTouchable(
        G4ThreeVector(100, 100, 100), G4ThreeVector(1, 0, 0), &touchable);

    std::ostringstream os;
    os << PrintableNavHistory{touchable.GetHistory()};
    EXPECT_EQ(
        R"({{pv='World0xdeadbeef_PV', lv=29='World0xdeadbeef'} -> {pv='env1', lv=28='Envelope'} -> {pv='Shape1', lv=27='Shape1'}})",
        os.str());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
