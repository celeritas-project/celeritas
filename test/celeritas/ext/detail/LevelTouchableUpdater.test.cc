//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/LevelTouchableUpdater.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/LevelTouchableUpdater.hh"

#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>

#include "corecel/cont/Span.hh"
#include "geocel/GeantGdmlLoader.hh"
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
struct TestResult
{
    std::vector<double> coords;
    std::vector<std::string> replicas;

    void print_expected() const;
};

void TestResult::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static double const expected_coords[] = "
         << repr(this->coords)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_coords, result.coords);\n"
            "static char const* const expected_replicas[] = "
         << repr(this->replicas)
         << ";\n"
            "EXPECT_VEC_EQ(expected_replicas, result.replicas);\n"
            "/*** END CODE ***/\n";
}

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
        auto* world_volume = ::celeritas::load_gdml(
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

        CELER_VALIDATE(names.size() < geo.max_depth() + 1,
                       << "input stack is too deep: " << names.size()
                       << " exceeds " << geo.max_depth());

        VecVI result;
        std::vector<std::string_view> missing;
        for (std::string_view sv : names)
        {
            auto vi = vol_inst.find_exact(Label::from_separator(sv));
            if (!vi)
            {
                missing.push_back(sv);
                continue;
            }
            result.push_back(vi);
        }
        CELER_VALIDATE(missing.empty(),
                       << "missing PVs from stack: "
                       << join(missing.begin(), missing.end(), ','));

        // Fill extra entries with empty volumes
        result.resize(geo.max_depth() + 1);
        return result;
    }

    G4VTouchable* touchable_history() { return touch_handle_(); }

    TestResult run(Span<IListSView const> names);

  private:
    G4TouchableHandle touch_handle_;
};

//---------------------------------------------------------------------------//
TestResult LevelTouchableUpdaterTest::run(Span<IListSView const> names)
{
    TestResult result;

    TouchableUpdater update = this->make_touchable_updater();
    auto* touch = this->touchable_history();

    for (IListSView level_names : names)
    {
        // Update the touchable
        auto vi_stack = this->find_vi_stack(level_names);
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
        result.coords.insert(result.coords.end(), {trans.x(), trans.y()});

        // Get the replica/copy numbers
        result.replicas.push_back([touch] {
            std::ostringstream os;
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

// See GeantGeoUtils.test.cc : MultiLevelTest.set_history
TEST_F(LevelTouchableUpdaterTest, out_of_order)
{
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

    auto result = run(all_level_names);

    static double const expected_coords[] = {
        -0,  -0,   -0,  -0,   -0,   -0,   100, 100, 125,  125, -75, 125,
        125, -125, 100, -100, -100, -100, 75,  75,  -125, 75,  125, 75,
        -75, 75,   -75, -125, -125, -75,  75,  -75, 125,  -75,
    };
    EXPECT_VEC_SOFT_EQ(expected_coords, result.coords);
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
    EXPECT_VEC_EQ(expected_replicas, result.replicas);
}

TEST_F(LevelTouchableUpdaterTest, all_points)
{
    static IListSView const all_level_names[] = {
        {"world_PV"},
        {"world_PV", "topsph1"},
        {"world_PV", "topbox1", "boxsph1@0"},
        {"world_PV", "topbox1"},
        {"world_PV", "topbox1", "boxtri@0"},
        {"world_PV", "topbox1", "boxsph2@0"},
        {"world_PV", "topbox2", "boxsph1@0"},
        {"world_PV", "topbox2"},
        {"world_PV", "topbox2", "boxtri@0"},
        {"world_PV", "topbox2", "boxsph2@0"},
        {"world_PV", "topbox4", "boxtri@1"},
        {"world_PV", "topbox4", "boxsph2@1"},
        {"world_PV", "topbox4", "boxsph1@1"},
        {"world_PV", "topbox4"},
        {"world_PV", "topbox3"},
        {"world_PV", "topbox3", "boxsph2@0"},
        {"world_PV", "topbox3", "boxsph1@0"},
        {"world_PV", "topbox3", "boxtri@0"},
    };

    auto result = run(all_level_names);

    static double const expected_coords[] = {
        -0,  -0,   -0,   -0,   125,  125,  100,  100, 125, 75,   75,   75,
        -75, 125,  -100, 100,  -75,  75,   -125, 75,  125, -75,  75,   -75,
        125, -125, 100,  -100, -100, -100, -125, -75, -75, -125, -125, -125,
    };
    EXPECT_VEC_SOFT_EQ(expected_coords, result.coords);
    static char const* const expected_replicas[] = {
        "0",
        "0,0",
        "31,21,0",
        "21,0",
        "1,21,0",
        "32,21,0",
        "31,22,0",
        "22,0",
        "1,22,0",
        "32,22,0",
        "1,24,0",
        "32,24,0",
        "31,24,0",
        "24,0",
        "23,0",
        "32,23,0",
        "31,23,0",
        "1,23,0",
    };
    EXPECT_VEC_EQ(expected_replicas, result.replicas);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
