//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/NaviTouchableUpdater.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/NaviTouchableUpdater.hh"

#include <cmath>
#include <G4Navigator.hh>
#include <G4TouchableHandle.hh>
#include <G4TouchableHistory.hh>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/g4/GeantGeoParams.hh"
#include "geocel/g4/GeantGeoTestBase.hh"
#include "celeritas/Units.hh"

#include "celeritas_test.hh"

using celeritas::test::from_cm;
using celeritas::test::ScopedLogStorer;

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Base class for navigation touchable updater.
 */
class NaviTouchableUpdaterBase : public ::celeritas::test::GeantGeoTestBase
{
  protected:
    using TouchableUpdater = NaviTouchableUpdater;

    void SetUp() override { touch_handle_ = new G4TouchableHistory; }

    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }

    G4LogicalVolume const* find_lv(std::string const& name) const
    {
        auto const& geo = *this->geometry();
        auto const* lv = geo.id_to_geant(geo.volumes().find_unique(name));
        CELER_ENSURE(lv);
        return lv;
    }

    TouchableUpdater make_touchable_updater()
    {
        auto const& geo = *this->geometry();
        return TouchableUpdater{
            std::make_shared<std::vector<G4LogicalVolume const*>>(),
            geo.world()};
    }

    G4VTouchable* touchable_history() { return touch_handle_(); }

  private:
    G4TouchableHandle touch_handle_;
};

//---------------------------------------------------------------------------//
/*!
 * Test with simple CMS geometry.
 *
 * | Radius [cm] | Volume name |
 * | ----------: | ----------- |
 * |          0  |             |
 * |         30  | vacuum_tube |
 * |        125  | si_tracker |
 * |        175  | em_calorimeter |
 * |        275  | had_calorimeter |
 * |        375  | sc_solenoid |
 * |        700  | fe_muon_chambers |
 * |             | world |
 */
class SimpleCmsNaviTest : public NaviTouchableUpdaterBase
{
  public:
    std::string geometry_basename() const final { return "simple-cms"; }
};

TEST_F(SimpleCmsNaviTest, correct)
{
    TouchableUpdater update = this->make_touchable_updater();
    auto update_cm = [&](Real3 const& pos_cm, std ::string lv_name) {
        return update(from_cm(pos_cm),
                      Real3{1, 0, 0},
                      this->find_lv(lv_name),
                      this->touchable_history());
    };

    EXPECT_TRUE(update_cm({15, 0, 0}, "vacuum_tube"));
    EXPECT_TRUE(update_cm({100, 0, 0}, "si_tracker"));
    EXPECT_TRUE(update_cm({150, 0, 0}, "em_calorimeter"));
}

TEST_F(SimpleCmsNaviTest, just_inside)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.5 * TouchableUpdater::max_quiet_step();
    auto const* tracker_lv = this->find_lv("si_tracker");
    auto const* calo_lv = this->find_lv("em_calorimeter");
    auto* th = this->touchable_history();

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    EXPECT_TRUE(update({from_cm(30) + eps, 0, 0}, {1, 0, 0}, tracker_lv, th));
    EXPECT_TRUE(update({from_cm(125) - eps, 0, 0}, {1, 0, 0}, tracker_lv, th));

    EXPECT_TRUE(update({from_cm(125) + eps, 0, 0}, {-1, 0, 0}, calo_lv, th));
    EXPECT_TRUE(update({from_cm(175) - eps, 0, 0}, {-1, 0, 0}, calo_lv, th));

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(SimpleCmsNaviTest, coincident)
{
    TouchableUpdater update = this->make_touchable_updater();
    auto* th = this->touchable_history();
    auto update_x = [&](real_type xpos, real_type xdir, char const* name) {
        return update({xpos, 0, 0}, {xdir, 0, 0}, this->find_lv(name), th);
    };

    // Coincident point should work in either volume, in or out
    real_type const r = from_cm(125);
    for (char const* lvname : {"si_tracker", "em_calorimeter"})
    {
        EXPECT_TRUE(update_x(r, 1, lvname));
        EXPECT_TRUE(update_x(r, -1, lvname));
    }
}

TEST_F(SimpleCmsNaviTest, coincident_tangent)
{
    TouchableUpdater update = this->make_touchable_updater();

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    // TODO: we can't seem to test the volume on the other side of an exact
    // coincident surface on a tangent
    real_type const r = from_cm(125);
    auto* th = this->touchable_history();
    EXPECT_FALSE(update({r, 0, 0}, {0, 1, 0}, this->find_lv("si_tracker"), th));
    EXPECT_TRUE(
        update({r, 0, 0}, {0, 1, 0}, this->find_lv("em_calorimeter"), th));

    static char const* const expected_log_messages[] = {
        R"(Failed to bump navigation state up to a distance of 1 [mm] at {1250, 0, 0} [mm] along {0, 1, 0} to try to reach "si_tracker"@0x0 (ID=1): found {{pv='em_calorimeter_pv', lv=2='em_calorimeter'}})",
    };
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(SimpleCmsNaviTest, just_outside_nowarn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.1 * TouchableUpdater::max_quiet_step();
    auto const* tracker_lv = this->find_lv("si_tracker");
    auto* th = this->touchable_history();
    auto update_x = [&](real_type xpos, real_type xdir) {
        return update({xpos, 0, 0}, {xdir, 0, 0}, tracker_lv, th);
    };

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    for (real_type xdir : {1.0, -1.0})
    {
        EXPECT_TRUE(update_x(from_cm(30) - eps, xdir));
        EXPECT_TRUE(update_x(from_cm(125) + 2 * eps, -xdir));
    }

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(SimpleCmsNaviTest, just_outside_warn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.1 * TouchableUpdater::max_step();
    auto const* tracker_lv = this->find_lv("si_tracker");
    auto* th = this->touchable_history();
    auto update_x = [&](real_type xpos, real_type xdir) {
        return update({xpos, 0, 0}, {xdir, 0, 0}, tracker_lv, th);
    };

    for (real_type xdir : {1.0, -1.0})
    {
        EXPECT_TRUE(update_x(from_cm(30) - eps, xdir));
        EXPECT_TRUE(update_x(from_cm(125) + eps, -xdir));
    }
}

TEST_F(SimpleCmsNaviTest, too_far)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 10 * TouchableUpdater::max_step();
    auto const* tracker_lv = this->find_lv("si_tracker");
    auto* th = this->touchable_history();
    auto update_x = [&](real_type xpos, real_type xdir) {
        return update({xpos, 0, 0}, {xdir, 0, 0}, tracker_lv, th);
    };

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    for (real_type xdir : {1.0, -1.0})
    {
        EXPECT_FALSE(update_x(from_cm(30) - eps, xdir));
        EXPECT_FALSE(update_x(from_cm(125) + eps, -xdir));
    }

    static char const* const expected_log_messages[] = {
        R"(Failed to bump navigation state up to a distance of 1 [mm] at {290, 0, 0} [mm] along {1, 0, 0} to try to reach "si_tracker"@0x0 (ID=1): found {{pv='vacuum_tube_pv', lv=0='vacuum_tube'}})",
        R"(Failed to bump navigation state up to a distance of 1 [mm] at {1260, 0, 0} [mm] along {-1, 0, 0} to try to reach "si_tracker"@0x0 (ID=1): found {{pv='em_calorimeter_pv', lv=2='em_calorimeter'}})",
        R"(Failed to bump navigation state up to a distance of 1 [mm] at {290, 0, 0} [mm] along {-1, 0, 0} to try to reach "si_tracker"@0x0 (ID=1): found {{pv='vacuum_tube_pv', lv=0='vacuum_tube'}})",
        R"(Failed to bump navigation state up to a distance of 1 [mm] at {1260, 0, 0} [mm] along {1, 0, 0} to try to reach "si_tracker"@0x0 (ID=1): found {{pv='em_calorimeter_pv', lv=2='em_calorimeter'}})",
    };
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[]
        = {"warning", "warning", "warning", "warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(SimpleCmsNaviTest, regression)
{
    using Real2 = Array<real_type, 2>;

    TouchableUpdater update = this->make_touchable_updater();
    auto* th = this->touchable_history();
    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    static struct
    {
        Real3 pos;  // [mm]
        Real3 dir;
        char const* volume;
    } const regressions[] = {
        {{-427.56983454727, 1174.5995217837, 747.90972779276},
         {-0.70886981480525, 0.21169894981561, 0.67282028826793},
         "em_calorimeter"},
        {{-180.84752203436, -1236.8514741857, 80.959574210285},
         {-0.34086888072834, 0.082800146878107, 0.9364574426144},
         "si_tracker"},
        {{128.83413807803, -270.82102012142, -2672.7505039643},
         {0.77015590259216, -0.30608417592167, -0.55961805095334},
         "si_tracker"},
        {{-206.25679395806, -217.74488354803, -954.9663190649},
         {0.61713971785822, -0.76637525189352, 0.17834669026092},
         "si_tracker"},
    };

    std::vector<real_type> radius;
    std::vector<real_type> ndot;

    for (auto const& v : regressions)
    {
        radius.push_back(std::hypot(v.pos[0], v.pos[1]));
        ndot.push_back(dot_product(make_unit_vector(Real2{v.pos[0], v.pos[1]}),
                                   Real2{v.dir[0], v.dir[1]}));
        EXPECT_TRUE(update(
            v.pos * units::millimeter, v.dir, this->find_lv(v.volume), th))
            << "from " << repr(v.pos) << " along " << repr(v.dir);
    }

    static double const expected_radius[] = {
        1249.9999999957,
        1250.002958165,
        299.90375135019,
        299.92448943893,
    };
    EXPECT_VEC_SOFT_EQ(expected_radius, radius);
    static double const expected_ndot[] = {
        0.4414022677194,
        -0.032612875869091,
        0.60724949202002,
        0.13198332160898,
    };
    EXPECT_VEC_SOFT_EQ(expected_ndot, ndot);
}

//---------------------------------------------------------------------------//
/*!
 * Test with multi-level geometry.
 *
 * See https://github.com/celeritas-project/g4vg/issues/16 , test point code is
 * from GeantGeo.test.cc MultiLevelTest.level_strings
 */
class MultiLevelNaviTest : public NaviTouchableUpdaterBase
{
  public:
    std::string geometry_basename() const final { return "multi-level"; }
};

TEST_F(MultiLevelNaviTest, all_points)
{
    static struct Inp
    {
        real_type x;
        real_type y;
        char const* lv;
    } lv_names[] = {
        {-5, 0, "world"},
        {0, 0, "sph"},
        {12.75, 12.75, "sph"},
        {7.25, 12.75, "box"},
        {12.75, 7.25, "tri"},
        {7.25, 7.25, "sph"},
        {-7.25, 12.75, "sph"},
        {-12.75, 12.75, "box"},
        {-7.25, 7.25, "tri"},
        {-12.75, 7.25, "sph"},
        {12.75, -7.25, "tri_refl"},
        {7.25, -7.25, "sph_refl"},
        {12.75, -12.75, "sph_refl"},
        {7.25, -12.75, "box_refl"},
        {-7.25, -7.25, "box"},
        {-12.75, -7.25, "sph"},
        {-7.25, -12.75, "sph"},
        {-12.75, -12.75, "tri"},
    };

    TouchableUpdater update = this->make_touchable_updater();
    auto* touch = this->touchable_history();
    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    std::vector<std::string> replicas;
    for (auto&& [x, y, lv_name] : lv_names)
    {
        update(from_cm(Real3{x, y, 0}),
               Real3{1, 0, 0},
               this->find_lv(lv_name),
               touch);

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
    EXPECT_VEC_EQ(expected_replicas, replicas);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
