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

TEST_F(TouchableUpdaterTest, just_inside_nowarn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 1e-6;  // below threshold

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    EXPECT_TRUE(
        update({125 - eps, 0, 0}, {1, 0, 0}, this->find_lv("si_tracker")));
    EXPECT_TRUE(
        update({175 - eps, 0, 0}, {1, 0, 0}, this->find_lv("em_calorimeter")));
    EXPECT_TRUE(
        update({125 + eps, 0, 0}, {-1, 0, 0}, this->find_lv("si_tracker")));
    EXPECT_TRUE(update(
        {175 + eps, 0, 0}, {-1, 0, 0}, this->find_lv("em_calorimeter")));

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(TouchableUpdaterTest, just_inside_warn)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 0.5 * celeritas::units::millimeter;

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    EXPECT_TRUE(
        update({125 - eps, 0, 0}, {1, 0, 0}, this->find_lv("si_tracker")));
    EXPECT_TRUE(update(
        {175 + eps, 0, 0}, {-1, 0, 0}, this->find_lv("em_calorimeter")));

    scoped_log_.print_expected();
}

TEST_F(TouchableUpdaterTest, just_outside)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = -1e-6;

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    EXPECT_TRUE(
        update({125 - eps, 0, 0}, {1, 0, 0}, this->find_lv("si_tracker")));
    EXPECT_TRUE(update(
        {175 + eps, 0, 0}, {-1, 0, 0}, this->find_lv("em_calorimeter")));

    scoped_log_.print_expected();
}

TEST_F(TouchableUpdaterTest, too_far)
{
    TouchableUpdater update = this->make_touchable_updater();
    real_type const eps = 2 * celeritas::units::millimeter;

    ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                LogLevel::diagnostic};

    EXPECT_TRUE(
        update({125 - eps, 0, 0}, {1, 0, 0}, this->find_lv("si_tracker")));
    EXPECT_TRUE(update(
        {175 + eps, 0, 0}, {-1, 0, 0}, this->find_lv("em_calorimeter")));

    scoped_log_.print_expected();
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
