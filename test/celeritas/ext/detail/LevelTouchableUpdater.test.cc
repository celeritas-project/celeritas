//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/LevelTouchableUpdater.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/LevelTouchableUpdater.hh"

#include <cstddef>
#include <iostream>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>

#include "corecel/cont/Span.hh"
#include "geocel/GeantGdmlLoader.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/InstancePathFinder.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/GlobalTestBase.hh"
#include "celeritas/OnlyCoreTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"
#include "celeritas/geo/CoreGeoParams.hh"

#include "AssertionHelper.hh"
#include "Comparer.hh"
#include "celeritas_test.hh"

using celeritas::test::InstancePathFinder;

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
struct TestResult
{
    static constexpr char class_str[] = "TestResult";

    using Dbl3 = Array<double, 3>;
    std::vector<Dbl3> coords;
    std::vector<std::string> replicas;

    friend std::ostream& operator<<(std::ostream& os, TestResult const& ref)
    {
        return os << '{' << CELER_ATTR_REPR(ref, coords) << ", "
                  << CELER_ATTR_REPR(ref, replicas) << '}';
    }

    friend bool check_ref_eq(::celeritas::test::AssertionHelper& helper,
                             TestResult const& expected,
                             TestResult const& actual)
    {
        using namespace ::celeritas::test;
        RefComparer<TestResult> compare{helper, expected, actual};
        CELER_INVOKE_ATTR(compare.is_soft_eq, TestResult, coords);
        CELER_INVOKE_ATTR(compare.is_eq, TestResult, replicas);
        return static_cast<bool>(compare);
    }
};

//---------------------------------------------------------------------------//
/*!
 * Test with multi-level geometry using "core" implementation.
 */
class LevelTouchableUpdaterTest : public ::celeritas::test::OnlyGeoTestBase
{
  protected:
    using TouchableUpdater = LevelTouchableUpdater;
    using IListSView = std::initializer_list<std::string_view>;

    void SetUp() override
    {
        touch_handle_ = new G4TouchableHistory;
        // Build geometry
        this->geometry();
    }

    TouchableUpdater make_touchable_updater()
    {
        auto ggeo = this->geant_geo();
        CELER_ASSERT(ggeo);
        return TouchableUpdater{std::move(ggeo)};
    }

    G4VTouchable* touchable_history() { return touch_handle_(); }

    TestResult run(Span<IListSView const> names);

  private:
    G4TouchableHandle touch_handle_;
};

TestResult LevelTouchableUpdaterTest::run(Span<IListSView const> names)
{
    TestResult result;

    CELER_ASSERT(this->volumes());
    InstancePathFinder find_vi_stack(*this->volumes());
    TouchableUpdater update = this->make_touchable_updater();
    auto* touch = this->touchable_history();

    for (IListSView level_names : names)
    {
        // Update the touchable
        auto vi_stack = find_vi_stack(level_names);
        try
        {
            EXPECT_TRUE(update(make_span(vi_stack), this->touchable_history()));
        }
        catch (celeritas::RuntimeError const& e)
        {
            ADD_FAILURE() << e.what();
            result.coords.insert(result.coords.end(), {0, 0});
            result.replicas.push_back(e.details().what);
            continue;
        }

        // Get the local-to-global x/y translation coordinates
        auto const& trans = touch->GetTranslation(0);
        if (touch->GetHistoryDepth() == 0 && touch->GetVolume() == nullptr)
        {
            // Special case: outside world
            result.coords.push_back({0, 0, 0});
        }
        else
        {
            result.coords.push_back({trans.x(), trans.y(), trans.z()});
        }

        // Get the replica/copy numbers
        result.replicas.push_back([touch] {
            std::ostringstream os;
            if (touch->GetHistoryDepth() == 0 && touch->GetVolume() == nullptr)
            {
                return std::string{};
            }
            os << touch->GetReplicaNumber(0);
            for (auto i : range(1, touch->GetHistoryDepth() + 1))
            {
                os << ',' << touch->GetReplicaNumber(i);
            }
            return std::move(os).str();
        }());
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Test with geometry that contains replicas.
 */
class ReplicaTest : public LevelTouchableUpdaterTest
{
    std::string_view gdml_basename() const override { return "replica"; }
};

TEST_F(ReplicaTest, all_points)
{
    static IListSView const all_level_names[] = {
        {"world_PV", "fSecondArmPhys"},
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
        {"world_PV"},
    };

    auto result = run(all_level_names);

    TestResult ref{{
                       {-2500, 0, 4330.1270189222},
                       {-4344.3747686898, 75, 5574.6778264911},
                       {-4604.1823898252, 75, 5424.6778264911},
                       {-3942.4038105677, 150, 6528.4437038563},
                       {-4587.0190528383, 150, 6444.9500548025},
                       {-4587.0190528383, -150, 6444.9500548025},
                       {-4552.211431703, 150, 6984.6614865054},
                       {-0, -0, -0},
                   },
                   {
                       "0,0",
                       "14,0,0,0",
                       "6,0,0,0",
                       "2,1,4,0,0,0",
                       "7,1,2,0,0,0",
                       "7,0,2,0,0,0",
                       "16,1,3,0,0,0",
                       "0",
                   }};
    EXPECT_REF_EQ(ref, result);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
