//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.test.cc
//---------------------------------------------------------------------------//
#include "accel/detail/HitManager.hh"

#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4NistManager.hh>
#include <G4Orb.hh>
#include <G4ParticleDefinition.hh>

#include "celeritas_config.h"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/ext/GeantGeoUtils.hh"
#include "celeritas/geo/GeoParams.hh"
#include "accel/SDTestBase.hh"
#include "accel/SetupOptions.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
class SimpleCmsTest : public ::celeritas::test::SDTestBase,
                      public ::celeritas::test::SimpleCmsTestBase
{
  protected:
    void SetUp() override
    {
        sd_setup_.enabled = true;
        sd_setup_.ignore_zero_deposition = false;
    }

    SPConstGeoI build_fresh_geometry(std::string_view basename) override
    {
        auto result = SDTestBase::build_fresh_geometry(basename);
        scoped_log_.clear();

        // Create unused volume when building geometry
        G4Material* mat
            = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
        SimpleCmsTest::detached_lv = new G4LogicalVolume(
            new G4Orb("unused_solid", 10.0), mat, "unused");

        return result;
    }

    SetStr detector_volumes() const final
    {
        // *Don't* add SD for si_tracker
        return {"em_calorimeter", "had_calorimeter"};
    }

    std::vector<std::string>
    volume_names(std::vector<VolumeId> const& vols) const
    {
        auto const& geo = *this->geometry();

        std::vector<std::string> result;
        for (VolumeId vid : vols)
        {
            result.push_back(geo.id_to_label(vid).name);
        }
        return result;
    }

    std::vector<std::string>
    particle_names(HitManager::VecParticle const& particles) const
    {
        std::vector<std::string> result;
        for (auto* par : particles)
        {
            CELER_ASSERT(par);
            result.push_back(par->GetParticleName());
        }
        return result;
    }

    HitManager make_hit_manager()
    {
        return HitManager(*this->geometry(), *this->particle(), sd_setup_, 1);
    }

  protected:
    SDSetupOptions sd_setup_;
    ::celeritas::test::ScopedLogStorer scoped_log_{&celeritas::world_logger()};
    static G4LogicalVolume* detached_lv;
};

G4LogicalVolume* SimpleCmsTest::detached_lv{nullptr};

TEST_F(SimpleCmsTest, no_change)
{
    HitManager man = this->make_hit_manager();

    EXPECT_EQ(0, man.geant_particles().size());
    EXPECT_EQ(2, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());
    static char const* const expected_vnames[]
        = {"em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
    if (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE)
    {
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    }
}

TEST_F(SimpleCmsTest, delete_one)
{
    // Create tracks for each particle type
    sd_setup_.track = true;

    sd_setup_.skip_volumes = find_geant_volumes({"had_calorimeter"});
    HitManager man = this->make_hit_manager();

    // Check volumes
    EXPECT_EQ(1, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());
    static char const* const expected_vnames[] = {"em_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);

    // Check particles
    auto pnames = this->particle_names(man.geant_particles());
    static std::string const expected_pnames[] = {"gamma", "e-", "e+"};
    EXPECT_VEC_EQ(expected_pnames, pnames);

    // Check log
    if (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE)
    {
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    }
}

TEST_F(SimpleCmsTest, add_duplicate)
{
    sd_setup_.force_volumes = find_geant_volumes({"em_calorimeter"});
    celeritas::world_logger().level(LogLevel::debug);
    HitManager man = this->make_hit_manager();
    celeritas::world_logger().level(Logger::default_level());

    EXPECT_EQ(2, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());

    static char const* const expected_vnames[]
        = {"em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
        static char const* const expected_log_messages[] = {
            "Mapped sensitive detector \"em_calorimeter\" on logical volume "
            "\"em_calorimeter\"@0x0 (ID=2) to VecGeom volume "
            "\"em_calorimeter@0x0\" (ID=2)",
            "Mapped sensitive detector \"had_calorimeter\" on logical volume "
            "\"had_calorimeter\"@0x0 (ID=3) to VecGeom volume "
            "\"had_calorimeter@0x0\" (ID=3)",
            "Ignored duplicate logical volume \"em_calorimeter\"@0x0 (ID=2)"};
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
        static char const* const expected_log_levels[]
            = {"debug", "debug", "debug"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
    }
}

TEST_F(SimpleCmsTest, add_one)
{
    sd_setup_.force_volumes = find_geant_volumes({"si_tracker"});
    HitManager man = this->make_hit_manager();

    EXPECT_EQ(3, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());

    static char const* const expected_vnames[]
        = {"si_tracker", "em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
    if (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE)
    {
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    }
}

TEST_F(SimpleCmsTest, no_detector)
{
    // No detectors
    sd_setup_.skip_volumes
        = find_geant_volumes({"em_calorimeter", "had_calorimeter"});
    EXPECT_THROW(this->make_hit_manager(), celeritas::RuntimeError);
    if (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_ORANGE)
    {
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    }
}

TEST_F(SimpleCmsTest, detached_detector)
{
    // Detector for LV that isn't in the world tree
    sd_setup_.skip_volumes = {};
    sd_setup_.force_volumes = {SimpleCmsTest::detached_lv};
    EXPECT_THROW(this->make_hit_manager(), celeritas::RuntimeError);

    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
        static char const* const expected_log_messages[]
            = {"Failed to find VecGeom volume corresponding to Geant4 volume "
               "\"unused_lv\"@0x0 (ID=7)"};
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
        static char const* const expected_log_levels[] = {"error"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
