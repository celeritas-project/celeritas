//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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

    G4VPhysicalVolume const* load_test_gdml(std::string_view basename)
    {
        return this->load(
            this->test_data_path("geocel", std::string(basename) + ".gdml"));
    }

    static void TearDownTestSuite()
    {
        loaded_filename_ = {};
        ::celeritas::reset_geant_geometry();
        world_volume_ = nullptr;
    }

  private:
    static std::string loaded_filename_;
    static G4VPhysicalVolume* world_volume_;
};

std::string ConverterTest::loaded_filename_{};
G4VPhysicalVolume* ConverterTest::world_volume_{nullptr};

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, testem3)
{
    Converter convert;
    auto result = convert(this->load_test_gdml("testem3")).input;

    ASSERT_EQ(2, result.universes.size());
    if (auto* unit = std::get_if<UnitInput>(&result.universes[0]))
    {
        SCOPED_TRACE("universe 0");
        EXPECT_EQ("World0x0", this->genericize_pointers(unit->label.name));
        EXPECT_EQ(53, unit->volumes.size());
        EXPECT_EQ(61, unit->surfaces.size());
        EXPECT_VEC_SOFT_EQ((Real3{-24, -24, -24}), unit->bbox.lower());
        EXPECT_VEC_SOFT_EQ((Real3{24, 24, 24}), unit->bbox.upper());
    }
    else
    {
        FAIL() << "wrong universe variant";
    }

    if (auto* unit = std::get_if<UnitInput>(&result.universes[1]))
    {
        SCOPED_TRACE("universe 1");
        EXPECT_EQ("Layer0x0", genericize_pointers(unit->label.name));
        EXPECT_EQ(4, unit->volumes.size());
        EXPECT_EQ(1, unit->surfaces.size());
        EXPECT_VEC_SOFT_EQ((Real3{-0.4, -20, -20}), unit->bbox.lower());
        EXPECT_VEC_SOFT_EQ((Real3{0.4, 20, 20}), unit->bbox.upper());
    }
    else
    {
        FAIL() << "wrong universe variant";
    }
}

TEST_F(ConverterTest, DISABLED_arbitrary)
{
    std::string filename = celeritas::getenv("GDML");
    CELER_VALIDATE(!filename.empty(),
                   << "Set the 'GDML' environment variable and run this "
                      "test with "
                      "--gtest_filter=*arbitrary "
                      "--gtest_also_run_disabled_tests");

    Converter convert([] {
        Converter::Options opts;
        opts.verbose = false;
        return opts;
    }());
    auto input = convert(this->load(filename)).input;

    filename += ".json";
    CELER_LOG(info) << "Writing JSON translation to " << filename;
    std::ofstream os(filename);
    os << input;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
