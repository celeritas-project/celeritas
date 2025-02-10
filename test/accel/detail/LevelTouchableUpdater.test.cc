//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/LevelTouchableUpdater.test.cc
//---------------------------------------------------------------------------//
#include "accel/detail/LevelTouchableUpdater.hh"

#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>

#include "corecel/cont/Span.hh"
#include "geocel/GeantGeoUtils.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/OnlyCoreTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"
#include "celeritas/geo/GeoParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test with multi-level geometry using "core" implementation.
 */
class LevelTouchableUpdaterTest : public ::celeritas::test::GlobalGeoTestBase,
                                  public ::celeritas::test::OnlyGeoTestBase,
                                  public ::celeritas::test::OnlyCoreTestBase
{
  protected:
    using TouchableUpdater = LevelTouchableUpdater;
    using IListSView = std::initializer_list<std::string_view>;
    using VecVI = std::vector<VolumeInstanceId>;

    void SetUp() override { touch_handle_ = new G4TouchableHistory; }

    std::string_view geometry_basename() const override
    {
        return "multi-level";
    }

    // We *must* build from a Geant4 geometry when using vecgeom/ORANGE:
    // otherwise PV pointers won't be set
    SPConstGeoI build_fresh_geometry(std::string_view basename) override
    {
        auto* world_volume = ::celeritas::load_geant_geometry_native(
            this->test_data_path("geocel", std::string(basename) + ".gdml"));

        return std::make_shared<GeoParams>(world_volume);
    }

    TouchableUpdater make_touchable_updater()
    {
        return TouchableUpdater{this->geometry()};
    }

    VecVI find_vi_stack(IListSView names) const
    {
        auto const& geo = *this->geometry();
        auto const& vol_inst = geo.volume_instances();

        // Pad the remaining depth with empty volumes
        VecVI result(geo.max_depth());
        auto name_iter = names.begin();
        for (auto i : range(names.size()))
        {
            result[i] = vol_inst.find_unique(std::string(*name_iter++));
            CELER_ASSERT(result[i]);
        }
        CELER_ASSERT(name_iter == names.end());

        return result;
    }

    G4VTouchable* touchable_history() { return touch_handle_(); }

  private:
    G4TouchableHandle touch_handle_;
};

// See GeantGeoUtils.test.cc : MultiLevelTest.set_history
TEST_F(LevelTouchableUpdaterTest, all)
{
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
    {
        GTEST_SKIP() << "ORANGE doesn't support volume instances";
    }

    static IListSView const all_level_names[] = {
        {"world_PV"},
        {"world_PV", "topsph1"},
        {"world_PV"},
        {"world_PV", "topbox1", "boxsph2"},
        {"world_PV", "topbox1"},
        {"world_PV", "topbox1", "boxsph1"},
        {"world_PV", "topbox2", "boxsph2"},
        {"world_PV", "topbox2", "boxsph1"},
        {"world_PV", "topbox4"},
        {"world_PV", "topbox3", "boxsph1"},
        {"world_PV", "topbox3", "boxsph2"},
    };

    TouchableUpdater update = this->make_touchable_updater();
    auto* touch = this->touchable_history();

    std::vector<double> coords;
    std::vector<std::string> replicas;

    for (IListSView level_names : all_level_names)
    {
        // Update the touchable
        auto vi_stack = this->find_vi_stack(level_names);
        EXPECT_TRUE(update(make_span(vi_stack), this->touchable_history()));

        // Get the local-to-global x/y translation coordinates
        auto const& trans = touch->GetTranslation(0);
        coords.insert(coords.end(), {trans.x(), trans.y()});

        // Get the replica/copy numbers
        replicas.push_back([touch] {
            std::ostringstream os;
            os << touch->GetReplicaNumber(0);
            for (auto i : range(1, touch->GetHistoryDepth() + 1))
            {
                os << ',' << touch->GetReplicaNumber(i);
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
}  // namespace detail
}  // namespace celeritas
