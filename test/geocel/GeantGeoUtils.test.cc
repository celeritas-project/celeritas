//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGeoUtils.test.cc
//---------------------------------------------------------------------------//
#include "geocel/GeantGeoUtils.hh"

#include <algorithm>
#include <initializer_list>
#include <string_view>
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
    using IListSView = std::initializer_list<std::string_view>;
    using VecPVConst = std::vector<G4VPhysicalVolume const*>;

    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }

    void SetUp() override
    {
        // Build geometry during setup
        ASSERT_TRUE(this->geometry());
    }

    VecPVConst find_pv_stack(IListSView names) const
    {
        auto const& geo = *this->geometry();
        auto const& vol_inst = geo.volume_instances();

        VecPVConst result;
        for (std::string_view sv : names)
        {
            auto vi = vol_inst.find_unique(std::string(sv));
            CELER_ASSERT(vi);
            result.push_back(geo.id_to_pv(vi));
            CELER_ASSERT(result.back());
        }
        return result;
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
class MultiLevelTest : public GeantGeoUtilsTest
{
    std::string geometry_basename() const override { return "multi-level"; }
};

TEST_F(MultiLevelTest, printable_nav)
{
    G4Navigator navi;
    G4TouchableHistory touchable;
    navi.SetWorldVolume(
        const_cast<G4VPhysicalVolume*>(this->geometry()->world()));
    navi.LocateGlobalPointAndUpdateTouchable(
        G4ThreeVector(75, -125, 0), G4ThreeVector(1, 0, 0), &touchable);

    std::ostringstream os;
    os << PrintableNavHistory{touchable.GetHistory()};
    EXPECT_EQ(R"({{pv='boxsph2', lv=26='sph'} -> {pv='topsph2', lv=27='box'}})",
              os.str());
}

//! Test set_history using some of the same properties that CMS HGcal needs
TEST_F(MultiLevelTest, set_history)
{
    static IListSView const all_level_names[] = {
        {"world_PV"},
        {"world_PV", "topsph1"},
        {"world_PV"},
        {"world_PV", "topbox1", "boxsph2"},
        {"world_PV", "topbox1"},
        {"world_PV", "topbox1", "boxsph1"},
        {"world_PV", "topbox2", "boxsph2"},
        {"world_PV", "topbox2", "boxsph1"},
        {"world_PV", "topsph2"},
        {"world_PV", "topbox3", "boxsph1"},
        {"world_PV", "topbox3", "boxsph2"},
    };

    G4TouchableHistory touch;
    G4NavigationHistory hist;
    std::vector<double> coords;
    std::vector<std::string> replicas;

    for (IListSView level_names : all_level_names)
    {
        auto phys_vols = this->find_pv_stack(level_names);
        CELER_ASSERT(phys_vols.size() == level_names.size());

        // Set the navigation history
        set_history(make_span(phys_vols), &hist);
        touch.UpdateYourself(hist.GetTopVolume(), &hist);

        // Get the local-to-global x/y translation coordinates
        auto const& trans = touch.GetTranslation(0);
        coords.insert(coords.end(), {trans.x(), trans.y()});

        // Get the replica/copy numbers
        replicas.push_back([&touch] {
            std::ostringstream os;
            os << touch.GetReplicaNumber(0);
            for (auto i : range(1, touch.GetHistoryDepth() + 1))
            {
                os << ',' << touch.GetReplicaNumber(i);
            }
            return std::move(os).str();
        }());
    }

    static double const expected_coords[] = {
        -0,  -0,   -0, -0,  -0,  -0,  75,   75,  100, 100,  125,
        125, -125, 75, -75, 125, 100, -100, -75, -75, -125, -125,
    };
    static char const* const expected_replicas[] = {
        "0",
        "11,0",
        "0",
        "32,21,0",
        "21,0",
        "31,21,0",
        "32,22,0",
        "31,22,0",
        "12,0",
        "31,23,0",
        "32,23,0",
    };

    EXPECT_VEC_SOFT_EQ(expected_coords, coords);
    EXPECT_VEC_EQ(expected_replicas, replicas);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
