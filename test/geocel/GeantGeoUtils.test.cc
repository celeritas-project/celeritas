//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "geocel/g4/Convert.hh"

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

auto labels_to_strings(std::vector<Label> const& labels)
{
    std::vector<std::string> result;
    for (auto lab : labels)
    {
        result.push_back(to_string(lab));
    }
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
        std::vector<std::string_view> missing;
        for (std::string_view sv : names)
        {
            auto vi = vol_inst.find_exact(Label::from_separator(sv));
            if (!vi)
            {
                missing.push_back(sv);
                continue;
            }
            result.push_back(geo.id_to_pv(vi));
            CELER_ASSERT(result.back());
        }
        CELER_VALIDATE(missing.empty(),
                       << "missing PVs from stack: "
                       << join(missing.begin(), missing.end(), ','));
        return result;
    }
};

//---------------------------------------------------------------------------//
class SolidsTest : public GeantGeoUtilsTest
{
    std::string geometry_basename() const override { return "solids"; }
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

TEST_F(SolidsTest, make_vol_labels)
{
    auto const& world = *this->geometry()->world();

    auto lv_str = labels_to_strings(make_logical_vol_labels(world));
    static char const* const expected_lv_str[] = {
        "box500",      "cone1",    "para1",     "sphere1",    "parabol1",
        "trap1",       "trd1",     "trd2",      "",           "trd3_refl@1",
        "tube100",     "boolean1", "polycone1", "genPocone1", "ellipsoid1",
        "tetrah1",     "orb1",     "polyhedr1", "hype1",      "elltube1",
        "ellcone1",    "arb8b",    "arb8a",     "xtru1",      "World",
        "trd3_refl@0",
    };
    EXPECT_VEC_EQ(expected_lv_str, lv_str);

    auto pv_str = labels_to_strings(make_physical_vol_labels(world));
    static char const* const expected_pv_str[] = {
        "box500_PV",   "cone1_PV",     "para1_PV",      "sphere1_PV",
        "parabol1_PV", "trap1_PV",     "trd1_PV",       "reflNormal",
        "reflected@0", "reflected@1",  "tube100_PV",    "boolean1_PV",
        "orb1_PV",     "polycone1_PV", "hype1_PV",      "polyhedr1_PV",
        "tetrah1_PV",  "arb8a_PV",     "arb8b_PV",      "ellipsoid1_PV",
        "elltube1_PV", "ellcone1_PV",  "genPocone1_PV", "xtru1_PV",
        "World_PV",
    };
    EXPECT_VEC_EQ(expected_pv_str, pv_str);
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

    auto get_nav_str = [&](Real3 const& pos) {
        navi.LocateGlobalPointAndUpdateTouchable(
            convert_to_geant(from_cm(pos), clhep_length),
            G4ThreeVector(1, 0, 0),
            &touchable);
        std::ostringstream os;
        os << PrintableNavHistory{touchable.GetHistory()};
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

//! Test set_history using some of the same properties that CMS HGcal needs
TEST_F(MultiLevelTest, set_history)
{
    // Note: the shuffled order is to check that we correctly update parent
    // levels even if we're in the same LV/PV
    static IListSView const all_level_names[] = {
        {"world_PV"},
        {"world_PV", "topsph1"},
        {"world_PV"},
        {"world_PV", "topbox1"},
        {"world_PV", "topbox1", "boxsph1@0"},
        {"world_PV", "topbox2", "boxsph1@0"},
        {"world_PV", "topbox4", "boxsph1@1"},
        {"world_PV", "topbox4"},
        {"world_PV", "topbox3"},
        {"world_PV", "topbox1", "boxsph2@0"},
        {"world_PV", "topbox2", "boxsph2@0"},
        {"world_PV", "topbox1", "boxtri@0"},
        {"world_PV", "topbox2", "boxtri@1"},
        {"world_PV", "topbox3", "boxsph1@0"},
        {"world_PV", "topbox3", "boxsph2@0"},
        {"world_PV", "topbox4", "boxsph2@1"},
        {"world_PV", "topbox4", "boxtri@1"},
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
        -0,  -0,   -0,  -0,   -0,   -0,   100, 100, 125,  125, -75, 125,
        125, -125, 100, -100, -100, -100, 75,  75,  -125, 75,  125, 75,
        -75, 75,   -75, -125, -125, -75,  75,  -75, 125,  -75,
    };
    static char const* const expected_replicas[] = {
        "0",
        "0,0",
        "0",
        "21,0",
        "31,21,0",
        "31,22,0",
        "31,24,0",
        "24,0",
        "23,0",
        "32,21,0",
        "32,22,0",
        "1,21,0",
        "1,22,0",
        "31,23,0",
        "32,23,0",
        "32,24,0",
        "1,24,0",
    };

    EXPECT_VEC_SOFT_EQ(expected_coords, coords);
    EXPECT_VEC_EQ(expected_replicas, replicas);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
