//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/TouchableUpdater.test.cc
//---------------------------------------------------------------------------//
#include "accel/detail/TouchableUpdater.hh"

#include <cmath>
#include <G4Navigator.hh>
#include <G4TouchableHistory.hh>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/GenericGeoTestBase.hh"
#include "celeritas/Units.hh"
#include "celeritas/ext/GeantGeoParams.hh"

#include "celeritas_test.hh"

using celeritas::test::ScopedLogStorer;

namespace celeritas
{
namespace detail
{
namespace test
{
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
class TouchableUpdaterTest : public ::celeritas::test::GenericGeantGeoTestBase
{
  protected:
    void SetUp() override
    {
        auto const& geo = *this->geometry();
        navi_.SetWorldVolume(const_cast<G4VPhysicalVolume*>(geo.world()));
        touch_handle_ = new G4TouchableHistory;
    }

    SPConstGeo build_geometry() final
    {
        return this->build_geometry_from_basename();
    }

    std::string geometry_basename() const override { return "simple-cms"; }

    G4LogicalVolume const* find_lv(std::string const& name) const
    {
        auto const& geo = *this->geometry();
        auto const* lv = geo.id_to_lv(geo.find_volume(name));
        CELER_ENSURE(lv);
        return lv;
    }

    TouchableUpdater make_touchable_updater()
    {
        return TouchableUpdater{&navi_, touch_handle_()};
    }

  private:
    G4Navigator navi_;
    G4TouchableHandle touch_handle_;
};

TEST_F(TouchableUpdaterTest, correct)
{
    TouchableUpdater update = this->make_touchable_updater();
    Real3 const dir{1, 0, 0};

    EXPECT_TRUE(update({15, 0, 0}, dir, this->find_lv("vacuum_tube")));
    EXPECT_TRUE(update({100, 0, 0}, dir, this->find_lv("si_tracker")));
    EXPECT_TRUE(update({150, 0, 0}, dir, this->find_lv("em_calorimeter")));
}

TEST_F(TouchableUpdaterTest, just_inside)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.5 * TouchableUpdater::max_quiet_step();
    auto const* tracker_lv = this->find_lv("si_tracker");
    auto const* calo_lv = this->find_lv("em_calorimeter");

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    EXPECT_TRUE(update({30 + eps, 0, 0}, {1, 0, 0}, tracker_lv));
    EXPECT_TRUE(update({125 - eps, 0, 0}, {1, 0, 0}, tracker_lv));

    EXPECT_TRUE(update({125 + eps, 0, 0}, {-1, 0, 0}, calo_lv));
    EXPECT_TRUE(update({175 - eps, 0, 0}, {-1, 0, 0}, calo_lv));

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(TouchableUpdaterTest, coincident)
{
    TouchableUpdater update = this->make_touchable_updater();
    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    // Coincident point should work in either volume, in or out
    for (char const* lv : {"si_tracker", "em_calorimeter"})
    {
        EXPECT_TRUE(update({125, 0, 0}, {1, 0, 0}, this->find_lv(lv)));
        EXPECT_TRUE(update({125, 0, 0}, {-1, 0, 0}, this->find_lv(lv)));
    }

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(TouchableUpdaterTest, coincident_tangent)
{
    TouchableUpdater update = this->make_touchable_updater();

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    // TODO: we can't seem to test the volume on the other side of an exact
    // coincident surface on a tangent
    EXPECT_FALSE(update({125, 0, 0}, {0, 1, 0}, this->find_lv("si_tracker")));
    EXPECT_TRUE(
        update({125, 0, 0}, {0, 1, 0}, this->find_lv("em_calorimeter")));

    static char const* const expected_log_messages[] = {
        "Failed to bump navigation state up to a distance of 1 [mm] at {1250, "
        "0, 0} [mm] along {0, 1, 0} to try to reach \"si_tracker\"@0x0 "
        "(ID=1): found {{pv='em_calorimeter_pv', lv=2='em_calorimeter'}}"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(TouchableUpdaterTest, just_outside_nowarn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.1 * TouchableUpdater::max_quiet_step();
    auto const* tracker_lv = this->find_lv("si_tracker");

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    for (auto& xdir : {1.0, -1.0})
    {
        EXPECT_TRUE(update({30 - eps, 0, 0}, {xdir, 0, 0}, tracker_lv));
        EXPECT_TRUE(update({125 + 2 * eps, 0, 0}, {-xdir, 0, 0}, tracker_lv));
    }

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(TouchableUpdaterTest, just_outside_warn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.1 * TouchableUpdater::max_step();
    auto const* tracker_lv = this->find_lv("si_tracker");

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    for (auto& xdir : {1.0, -1.0})
    {
        EXPECT_TRUE(update({30 - eps, 0, 0}, {xdir, 0, 0}, tracker_lv));
        EXPECT_TRUE(update({125 + eps, 0, 0}, {-xdir, 0, 0}, tracker_lv));
    }

    static char const* const expected_log_messages[]
        = {"Bumping navigation state by 0.10000000000003 [mm] at {299.9, 0, "
           "0} [mm] along {1, 0, 0} from {{pv='vacuum_tube_pv', "
           "lv=0='vacuum_tube'}} to try to reach \"si_tracker\"@0x0 (ID=1)",
           "...bumped to {{pv='si_tracker_pv', lv=1='si_tracker'}}",
           "Bumping navigation state by 0.1000000000001 [mm] at {1250.1, 0, "
           "0} [mm] along {-1, 0, 0} from {{pv='em_calorimeter_pv', "
           "lv=2='em_calorimeter'}} to try to reach \"si_tracker\"@0x0 (ID=1)",
           "...bumped to {{pv='si_tracker_pv', lv=1='si_tracker'}}",
           "Bumping navigation state by 0.10000000000003 [mm] at {299.9, 0, "
           "0} [mm] along {1, -0, -0} from {{pv='vacuum_tube_pv', "
           "lv=0='vacuum_tube'}} to try to reach \"si_tracker\"@0x0 (ID=1)",
           "...bumped to {{pv='si_tracker_pv', lv=1='si_tracker'}}",
           "Bumping navigation state by 0.1000000000001 [mm] at {1250.1, 0, "
           "0} [mm] along {-1, -0, -0} from {{pv='em_calorimeter_pv', "
           "lv=2='em_calorimeter'}} to try to reach \"si_tracker\"@0x0 (ID=1)",
           "...bumped to {{pv='si_tracker_pv', lv=1='si_tracker'}}"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"warning",
                                                      "diagnostic",
                                                      "warning",
                                                      "diagnostic",
                                                      "warning",
                                                      "diagnostic",
                                                      "warning",
                                                      "diagnostic"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(TouchableUpdaterTest, too_far)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 10 * TouchableUpdater::max_step();
    auto const* tracker_lv = this->find_lv("si_tracker");

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    for (auto& xdir : {1.0, -1.0})
    {
        EXPECT_FALSE(update({30 - eps, 0, 0}, {xdir, 0, 0}, tracker_lv));
        EXPECT_FALSE(update({125 + eps, 0, 0}, {-xdir, 0, 0}, tracker_lv));
    }

    static char const* const expected_log_messages[] = {
        "Failed to bump navigation state up to a distance of 1 [mm] at {290, "
        "0, 0} [mm] along {1, 0, 0} to try to reach \"si_tracker\"@0x0 "
        "(ID=1): found {{pv='vacuum_tube_pv', lv=0='vacuum_tube'}}",
        "Failed to bump navigation state up to a distance of 1 [mm] at {1260, "
        "0, 0} [mm] along {-1, 0, 0} to try to reach \"si_tracker\"@0x0 "
        "(ID=1): found {{pv='em_calorimeter_pv', lv=2='em_calorimeter'}}",
        "Failed to bump navigation state up to a distance of 1 [mm] at {290, "
        "0, 0} [mm] along {-1, 0, 0} to try to reach \"si_tracker\"@0x0 "
        "(ID=1): found {{pv='vacuum_tube_pv', lv=0='vacuum_tube'}}",
        "Failed to bump navigation state up to a distance of 1 [mm] at {1260, "
        "0, 0} [mm] along {1, 0, 0} to try to reach \"si_tracker\"@0x0 "
        "(ID=1): found {{pv='em_calorimeter_pv', lv=2='em_calorimeter'}}"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[]
        = {"warning", "warning", "warning", "warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(TouchableUpdaterTest, regression)
{
    using Real2 = Array<real_type, 2>;

    TouchableUpdater update = this->make_touchable_updater();
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
        // NOTE: Geant4 computes an infinite distance in both directions here
        //        {{-1175.9420796891, 424.47404011144, 747.17772704325},
        //         {0.46254766852101, 0.83983515631001, -0.28412420624001},
        //         "si_tracker"},
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
        EXPECT_TRUE(
            update(v.pos * units::millimeter, v.dir, this->find_lv(v.volume)))
            << "from " << repr(v.pos) << " along " << repr(v.dir);
    }

    static double const expected_radius[] = {
        1249.9999999957,
        // 1250.2071770359,
        1250.002958165,
        299.90375135019,
        299.92448943893,
    };
    EXPECT_VEC_SOFT_EQ(expected_radius, radius);
    static double const expected_ndot[] = {
        0.4414022677194,
        // -0.14992798705076,
        -0.032612875869091,
        0.60724949202002,
        0.13198332160898,
    };
    EXPECT_VEC_SOFT_EQ(expected_ndot, ndot);

    static char const* const expected_log_messages[]
        = {"Bumping navigation state by 0.09071774570126 [mm] at "
           "{-180.84752203436, -1236.8514741857, 80.959574210285} [mm] along "
           "{-0.34086888072834, 0.082800146878107, 0.9364574426144} from "
           "{{pv='em_calorimeter_pv', lv=2='em_calorimeter'}} to try to reach "
           "\"si_tracker\"@0x0 (ID=1)",
           "...bumped to {{pv='si_tracker_pv', lv=1='si_tracker'}}",
           "Bumping navigation state by 0.15847742469601 [mm] at "
           "{128.83413807803, -270.82102012142, -2672.7505039643} [mm] along "
           "{0.77015590259216, -0.30608417592167, -0.55961805095334} from "
           "{{pv='vacuum_tube_pv', lv=0='vacuum_tube'}} to try to reach "
           "\"si_tracker\"@0x0 (ID=1)",
           "...bumped to {{pv='si_tracker_pv', lv=1='si_tracker'}}",
           "Bumping navigation state by 0.56824514959388 [mm] at "
           "{-206.25679395806, -217.74488354803, -954.9663190649} [mm] along "
           "{0.61713971785822, -0.76637525189352, 0.17834669026092} from "
           "{{pv='vacuum_tube_pv', lv=0='vacuum_tube'}} to try to reach "
           "\"si_tracker\"@0x0 (ID=1)",
           "...bumped to {{pv='si_tracker_pv', lv=1='si_tracker'}}"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"warning",
                                                      "diagnostic",
                                                      "warning",
                                                      "diagnostic",
                                                      "warning",
                                                      "diagnostic"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
