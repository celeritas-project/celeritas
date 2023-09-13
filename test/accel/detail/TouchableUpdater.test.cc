//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/TouchableUpdater.test.cc
//---------------------------------------------------------------------------//
#include "accel/detail/TouchableUpdater.hh"

#include <G4Navigator.hh>
#include <G4TouchableHistory.hh>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
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

TEST_F(TouchableUpdaterTest, just_outside_nowarn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.1 * TouchableUpdater::max_quiet_step()
                          * celeritas::units::millimeter;
    auto const* tracker_lv = this->find_lv("si_tracker");

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    for (auto& xdir : {1.0, -1.0})
    {
        EXPECT_TRUE(update({30 - eps, 0, 0}, {xdir, 0, 0}, tracker_lv));
        EXPECT_TRUE(update({125 + eps, 0, 0}, {-xdir, 0, 0}, tracker_lv));
    }

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(TouchableUpdaterTest, just_outside_warn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.1 * TouchableUpdater::max_step()
                          * celeritas::units::millimeter;
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
    real_type const eps = 10 * TouchableUpdater::max_step()
                          * celeritas::units::millimeter;
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

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
