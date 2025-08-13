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
#include <G4NavigationHistory.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHistory.hh>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeoParamsInterface.hh"
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

//---------------------------------------------------------------------------//
}  // namespace

class GeantGeoUtilsTest : public GeantGeoTestBase
{
  public:
    using IListSView = std::initializer_list<std::string_view>;
    using VecPhysInst = std::vector<GeantPhysicalInstance>;
    using ReplicaId = GeantPhysicalInstance::ReplicaId;

    void SetUp() override
    {
        // Build geometry during setup
        ASSERT_TRUE(this->geometry());

        // Clear all copy numbers
        for (auto* pv : *G4PhysicalVolumeStore::GetInstance())
        {
            if (pv && pv->VolumeType() != EVolume::kNormal)
            {
                pv->SetCopyNo(0);
            }
        }
    }

    VecPhysInst find_pv_stack(IListSView names) const
    {
        auto const& geo = *this->geometry();
        auto const& vol_inst = geo.volume_instances();

        VecPhysInst result;
        std::vector<std::string_view> missing;
        for (std::string_view sv : names)
        {
            auto lab = Label::from_separator(sv);
            ReplicaId replica;
            if (auto pos = lab.ext.find('+'); pos != std::string::npos)
            {
                // Convert replica ID and erase remainder
                replica
                    = id_cast<ReplicaId>(std::stoi(lab.ext.substr(pos + 1)));
                lab.ext.erase(pos, std::string::npos);
            }

            auto vi = vol_inst.find_exact(lab);
            if (!vi)
            {
                missing.push_back(sv);
                continue;
            }

            auto phys_inst = geo.id_to_geant(vi);
            if (replica)
            {
                phys_inst.replica = replica;
            }
            result.push_back(phys_inst);
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
    auto geo = this->make_geo_track_view();
    auto get_nav_str = [&geo](Real3 const& pos) {
        geo = {from_cm(pos), Real3{1, 0, 0}};
        std::ostringstream os;
        os << PrintableNavHistory{geo.nav_history()};
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
        {"world_PV"},
        {},
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
        if (touch.GetHistoryDepth() == 0 && touch.GetVolume() == nullptr)
        {
            // Special case: outside world
            coords.insert(coords.end(), {0, 0});
        }
        else
        {
            auto const& trans = touch.GetTranslation(0);
            coords.insert(coords.end(), {trans.x(), trans.y()});
        }

        // Get the replica/copy numbers
        replicas.push_back([&touch] {
            if (touch.GetHistoryDepth() == 0 && touch.GetVolume() == nullptr)
            {
                return std::string{};
            }
            std::ostringstream os;
            for (auto i : range(touch.GetHistoryDepth() + 1))
            {
                if (i != 0)
                {
                    os << ',';
                }
                os << touch.GetReplicaNumber(i);
            }
            return std::move(os).str();
        }());
    }

    static double const expected_coords[] = {
        -0,   -0,   -0,   -0,   -0,   -0,  100, 100,  125, 125, -75, 125, 125,
        -125, 100,  -100, -100, -100, 75,  75,  -125, 75,  125, 75,  -75, 75,
        -75,  -125, -125, -75,  75,   -75, 125, -75,  0,   0,   0,   0};
    static char const* const expected_replicas[] = {
        "0",       "0,0",     "0",      "21,0",    "31,21,0",
        "31,22,0", "31,24,0", "24,0",   "23,0",    "32,21,0",
        "32,22,0", "1,21,0",  "1,22,0", "31,23,0", "32,23,0",
        "32,24,0", "1,24,0",  "0",      "",
    };

    EXPECT_VEC_SOFT_EQ(expected_coords, coords);
    EXPECT_VEC_EQ(expected_replicas, replicas);
}

//---------------------------------------------------------------------------//
class ReplicaTest : public GeantGeoUtilsTest
{
    std::string_view gdml_basename() const override { return "replica"; }
};

TEST_F(ReplicaTest, is_replica)
{
    auto track = this->make_geo_track_view();
    auto get_replicas = [&track](Real3 const& pos) {
        track = {from_cm(pos), Real3{0, 0, 1}};

        auto* hist = track.nav_history();
        CELER_ASSERT(hist);

        std::vector<std::string> replicas;
        for (auto i : range(hist->GetDepth() + 1))
        {
            auto* pv = hist->GetVolume(i);
            if (!pv)
            {
                replicas.push_back("<null>");
                continue;
            }
            if (is_replica(*pv))
            {
                replicas.push_back(pv->GetName());
            }
        }
        return replicas;
    };

    {
        static char const* const expected[]
            = {"HadCalColumn_PV", "HadCalCell_PV", "HadCalLayer_PV"};
        auto actual = get_replicas({-400, 0.1, 650});
        EXPECT_VEC_EQ(expected, actual);
    }
    {
        static char const* const expected[]
            = {"HadCalColumn_PV", "HadCalCell_PV", "HadCalLayer_PV"};
        auto actual = get_replicas({-450, 0.1, 650});
        EXPECT_VEC_EQ(expected, actual);
    }
    {
        static char const* const expected[]
            = {"HadCalColumn_PV", "HadCalCell_PV", "HadCalLayer_PV"};
        auto actual = get_replicas({-450, 0.1, 700});
        EXPECT_VEC_EQ(expected, actual);
    }
}

//! Test set_history using some of the same properties that CMS HGcal needs
TEST_F(ReplicaTest, set_history)
{
    // Note: the shuffled order is to check that we correctly update parent
    // levels even if we're in the same LV/PV
    static IListSView const all_level_names[] = {
        {"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@+14"},
        {"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@+6"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@+4",
         "HadCalCell_PV@+1",
         "HadCalLayer_PV@+2"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@+2",
         "HadCalCell_PV@+1",
         "HadCalLayer_PV@+7"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@+2",
         "HadCalCell_PV@+0",
         "HadCalLayer_PV@+7"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@+3",
         "HadCalCell_PV@+1",
         "HadCalLayer_PV@+16"},
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
        coords.insert(coords.end(), {trans.x(), trans.y(), trans.z()});

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
        -4344.3747686898,
        75,
        5574.6778264911,
        -4604.1823898252,
        75,
        5424.6778264911,
        -3942.4038105677,
        150,
        6528.4437038563,
        -4587.0190528383,
        150,
        6444.9500548025,
        -4587.0190528383,
        -150,
        6444.9500548025,
        -4552.211431703,
        150,
        6984.6614865054,
    };
    static char const* const expected_replicas[] = {
        "14,0,0,0",
        "6,0,0,0",
        "2,1,4,0,0,0",
        "7,1,2,0,0,0",
        "7,0,2,0,0,0",
        "16,1,3,0,0,0",
    };

    EXPECT_VEC_SOFT_EQ(expected_coords, coords);
    EXPECT_VEC_EQ(expected_replicas, replicas);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
