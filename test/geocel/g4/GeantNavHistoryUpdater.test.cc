//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantNavHistoryUpdater.test.cc
//---------------------------------------------------------------------------//
#include "geocel/g4/GeantNavHistoryUpdater.hh"

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <string_view>
#include <G4LogicalVolume.hh>
#include <G4NavigationHistory.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4ThreeVector.hh>
#include <G4TouchableHistory.hh>

#include "geocel/InstancePathFinder.hh"
#include "geocel/g4/GeantGeoTestBase.hh"

#include "celeritas_test.hh"

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

class GeantNavHistoryUpdaterTest : public GeantGeoTestBase
{
  public:
    using IListSView = std::initializer_list<std::string_view>;

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
};

//---------------------------------------------------------------------------//
class MultiLevelTest : public GeantNavHistoryUpdaterTest
{
    std::string_view gdml_basename() const override { return "multi-level"; }
};

//! Test history setter using some of the same properties that CMS HGcal needs
TEST_F(MultiLevelTest, history_updater)
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

    GeantNavHistoryUpdater set_history(*this->geant_geo());
    InstancePathFinder find_vi_stack(*this->volumes());

    G4TouchableHistory touch;
    G4NavigationHistory hist;
    std::vector<double> coords;
    std::vector<std::string> replicas;

    for (IListSView level_names : all_level_names)
    {
        auto phys_vols = find_vi_stack(level_names);
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
class ReplicaTest : public GeantNavHistoryUpdaterTest
{
    std::string_view gdml_basename() const override { return "replica"; }
};

//! Test set_history using some of the same properties that CMS HGcal needs
TEST_F(ReplicaTest, history_updater)
{
    // Note: the shuffled order is to check that we correctly update parent
    // levels even if we're in the same LV/PV
    static IListSView const all_level_names[] = {
        {"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@14"},
        {"world_PV", "fSecondArmPhys", "EMcalorimeter", "cell_param@6"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@4",
         "HadCalCell_PV@1",
         "HadCalLayer_PV@2"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@2",
         "HadCalCell_PV@1",
         "HadCalLayer_PV@7"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@2",
         "HadCalCell_PV@0",
         "HadCalLayer_PV@7"},
        {"world_PV",
         "fSecondArmPhys",
         "HadCalorimeter",
         "HadCalColumn_PV@3",
         "HadCalCell_PV@1",
         "HadCalLayer_PV@16"},
    };

    GeantNavHistoryUpdater set_history(*this->geant_geo());
    InstancePathFinder find_vi_stack(*this->volumes());

    G4TouchableHistory touch;
    G4NavigationHistory hist;
    std::vector<double> coords;
    std::vector<std::string> replicas;

    for (IListSView level_names : all_level_names)
    {
        auto phys_vols = find_vi_stack(level_names);
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
