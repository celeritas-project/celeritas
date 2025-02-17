//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeantGdmlLoader.test.cc
//---------------------------------------------------------------------------//
#include "geocel/GeantGdmlLoader.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class GeantGdmlLoaderTest : public ::celeritas::test::Test
{
  protected:
    using PT = GeantGdmlLoader::PointerTreatment;
    using SetLV = std::unordered_set<G4LogicalVolume const*>;

    std::string gdml_path(std::string const& basename) const
    {
        return this->test_data_path("geocel", basename + std::string{".gdml"});
    }

    void TearDown() override { ::celeritas::reset_geant_geometry(); }

    ScopedLogStorer scoped_log_{&celeritas::world_logger(), LogLevel::warning};
};

//---------------------------------------------------------------------------//

using SolidsTest = GeantGdmlLoaderTest;

TEST_F(SolidsTest, load_save)
{
    ScopedLogStorer scoped_log_local_{&celeritas::self_logger(),
                                      LogLevel::warning};
    auto* world = load_gdml(this->gdml_path("solids"));
    static char const* const expected_local_log_levels[] = {"error"};
    EXPECT_VEC_EQ(expected_local_log_levels, scoped_log_local_.levels());
    scoped_log_local_ = {};

    ASSERT_TRUE(world);

    auto vols = find_geant_volumes({"box500", "trd3", "trd1"});

    save_gdml(world, this->make_unique_filename(".gdml"));

    static char const* const expected_log_messages[] = {
        R"(Geant4 regions have not been set up: skipping export of energy cuts and regions)"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

//---------------------------------------------------------------------------//

using SimpleCmsTest = GeantGdmlLoaderTest;

TEST_F(SimpleCmsTest, detectors)
{
    GeantGdmlLoader::Options opts;
    opts.detectors = true;

    GeantGdmlLoader load_gdml(opts);
    auto loaded = load_gdml(this->gdml_path("simple-cms"));
    EXPECT_TRUE(loaded.world);
    EXPECT_EQ(1, loaded.detectors.count("si_tracker_sd"));
    EXPECT_EQ(1, loaded.detectors.count("em_calorimeter_sd"));

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

//---------------------------------------------------------------------------//

using CmsEeTest = GeantGdmlLoaderTest;

TEST_F(CmsEeTest, ignore)
{
    GeantGdmlLoader::Options opts;
    opts.detectors = true;
    opts.pointers = PT::ignore;

    GeantGdmlLoader load_gdml(opts);
    auto loaded = load_gdml(this->gdml_path("cms-ee-back-dee"));

    EXPECT_EQ(2, loaded.detectors.count("ee_back_plate"));
    EXPECT_EQ(2, loaded.detectors.count("ee_s_ring"));

    // Reflected volume name is intact
    SetLV found;
    EXPECT_NO_THROW(found = find_geant_volumes({"EEBackQuad0x7f4a8f07c900",
                                                "EEBackQuad0x7f4a8f07c900_"
                                                "refl"}));
    EXPECT_EQ(2, found.size());
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;

    save_gdml(loaded.world, this->make_unique_filename(".gdml"));
}

TEST_F(CmsEeTest, truncate)
{
    GeantGdmlLoader::Options opts;
    opts.detectors = true;
    opts.pointers = PT::truncate;

    GeantGdmlLoader load_gdml(opts);
    auto loaded = load_gdml(this->gdml_path("cms-ee-back-dee"));

    EXPECT_EQ(2, loaded.detectors.count("ee_back_plate"));
    EXPECT_EQ(2, loaded.detectors.count("ee_s_ring"));

    // Reflected volume name is deleted by G4 GDML parser
    SetLV found;
    EXPECT_NO_THROW(found = find_geant_volumes({"EEBackQuad"}));
    EXPECT_EQ(2, found.size());

    static char const* const expected_log_messages[]
        = {"Multiple Geant4 volumes are mapped to name 'EEBackQuad'"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"warning"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());

    save_gdml(loaded.world, this->make_unique_filename(".gdml"));
}

TEST_F(CmsEeTest, remove)
{
    GeantGdmlLoader::Options opts;
    opts.detectors = true;
    opts.pointers = PT::remove;

    GeantGdmlLoader load_gdml(opts);
    auto loaded = load_gdml(this->gdml_path("cms-ee-back-dee"));

    EXPECT_EQ(2, loaded.detectors.count("ee_back_plate"));
    EXPECT_EQ(2, loaded.detectors.count("ee_s_ring"));

    // Pointer is removed but reflected volume suffix is retained
    SetLV found;
    EXPECT_NO_THROW(
        found = find_geant_volumes({"EEBackQuad", "EEBackQuad_refl"}));
    EXPECT_EQ(2, found.size());

    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;

    save_gdml(loaded.world, this->make_unique_filename(".gdml"));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
