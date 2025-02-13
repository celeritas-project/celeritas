//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Converter.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/Converter.hh"

#include <fstream>

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/UnitUtils.hh"
#include "orange/OrangeInput.hh"

#include "celeritas_test.hh"

using namespace celeritas::test;

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//

class ConverterTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override { verbose_ = !celeritas::getenv("VERBOSE").empty(); }

    //! Make a converter
    Converter make_converter(std::string_view filename = {})
    {
        Converter::Options opts;
        opts.verbose = verbose_;
        if (!filename.empty())
        {
            opts.proto_output_file = std::string(filename) + ".protos.json";
            opts.debug_output_file = std::string(filename) + ".csg.json";
        }
        return Converter{std::move(opts)};
    };

    //! Helper function: build via Geant4 GDML reader
    G4VPhysicalVolume const* load(std::string const& filename)
    {
        CELER_EXPECT(!filename.empty());
        if (filename == loaded_filename_)
        {
            return world_volume_;
        }

        if (world_volume_)
        {
            // Clear old geant4 data
            TearDownTestSuite();
        }
        ScopedLogStorer scoped_log_{&celeritas::self_logger(),
                                    LogLevel::warning};
        world_volume_ = ::celeritas::load_geant_geometry_native(filename);
        EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
        loaded_filename_ = filename;

        return world_volume_;
    }

    //! Load a test input
    G4VPhysicalVolume const* load_test_gdml(std::string_view basename)
    {
        return this->load(
            this->test_data_path("geocel", std::string(basename) + ".gdml"));
    }

    //! Save ORANGE output
    void write_org_json(OrangeInput const& inp, std::string const& filename)
    {
        if (!verbose_)
        {
            return;
        }
        auto out_filename = filename + ".org.json";
        CELER_LOG(info) << "Writing JSON translation to " << out_filename;
        std::ofstream os(out_filename);
        os << inp;
    }

    static void TearDownTestSuite()
    {
        loaded_filename_ = {};
        ::celeritas::reset_geant_geometry();
        world_volume_ = nullptr;
    }

    bool verbose_{false};

  private:
    static std::string loaded_filename_;
    static G4VPhysicalVolume* world_volume_;
};

std::string ConverterTest::loaded_filename_{};
G4VPhysicalVolume* ConverterTest::world_volume_{nullptr};

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, testem3)
{
    std::string const basename = "testem3";
    auto convert = this->make_converter(basename);
    auto result = convert(this->load_test_gdml(basename)).input;
    write_org_json(result, basename);

    ASSERT_EQ(2, result.universes.size());
    if (auto* unit = std::get_if<UnitInput>(&result.universes[0]))
    {
        SCOPED_TRACE("universe 0");
        EXPECT_EQ(Label{"world"}, unit->label);
        EXPECT_EQ(53, unit->volumes.size());
        EXPECT_EQ(61, unit->surfaces.size());
        EXPECT_VEC_SOFT_EQ((Real3{-24, -24, -24}), to_cm(unit->bbox.lower()));
        EXPECT_VEC_SOFT_EQ((Real3{24, 24, 24}), to_cm(unit->bbox.upper()));
    }
    else
    {
        FAIL() << "wrong universe variant";
    }

    if (auto* unit = std::get_if<UnitInput>(&result.universes[1]))
    {
        SCOPED_TRACE("universe 1");
        EXPECT_EQ(Label{"layer"}, unit->label);
        EXPECT_EQ(4, unit->volumes.size());
        EXPECT_EQ(1, unit->surfaces.size());
        EXPECT_VEC_SOFT_EQ((Real3{-0.4, -20, -20}), to_cm(unit->bbox.lower()));
        EXPECT_VEC_SOFT_EQ((Real3{0.4, 20, 20}), to_cm(unit->bbox.upper()));
    }
    else
    {
        FAIL() << "wrong universe variant";
    }
}

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, tilecal_plug)
{
    std::string const basename = "tilecal-plug";
    auto convert = this->make_converter(basename);
    auto result = convert(this->load_test_gdml(basename)).input;
    write_org_json(result, basename);

    ASSERT_EQ(1, result.universes.size());
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        ASSERT_EQ(4, unit.volumes.size());
        EXPECT_EQ("Tile_Plug1Module", unit.volumes[1].label.name);
        EXPECT_EQ("Tile_Absorber", unit.volumes[2].label.name);
        EXPECT_EQ("Tile_ITCModule", unit.volumes[3].label.name);
    }
}

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, DISABLED_arbitrary)
{
    verbose_ = true;
    std::string filename = celeritas::getenv("GDML");
    CELER_VALIDATE(!filename.empty(),
                   << "Set the 'GDML' environment variable and run this "
                      "test with "
                      "--gtest_filter=*arbitrary "
                      "--gtest_also_run_disabled_tests");

    auto convert = this->make_converter(filename);
    auto input = convert(this->load(filename)).input;

    write_org_json(input, filename);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
