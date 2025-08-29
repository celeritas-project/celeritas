//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Converter.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/Converter.hh"

#include <fstream>

#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "geocel/UnitUtils.hh"
#include "orange/OrangeInput.hh"

#include "GeantLoadTestBase.hh"
#include "celeritas_test.hh"

using namespace celeritas::test;

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//

class ConverterTest : public GeantLoadTestBase
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

    bool verbose_{false};
};

struct VolumeInstanceAccessor
{
    std::vector<VolumeInput> const& volumes;

    std::string operator()(LocalVolumeId lv_id) const
    {
        if (!lv_id)
            return "<null lv>";
        return (*this)(lv_id.get());
    }

    std::string operator()(size_type i) const
    {
        if (i >= volumes.size())
        {
            return "<out of bounds>";
        }
        auto const& var_label = volumes[i].label;
        if (auto* vi_id = std::get_if<VolumeInstanceId>(&var_label))
        {
            if (*vi_id)
            {
                return std::to_string(vi_id->get());
            }
            return "<null>";
        }
        CELER_ASSUME(std::holds_alternative<Label>(var_label));
        return to_string(std::get<Label>(var_label));
    }
};

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, lar_sphere)
{
    verbose_ = true;
    std::string const basename = "lar-sphere";
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
    write_org_json(result, basename);

    ASSERT_EQ(1, result.universes.size());
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        EXPECT_EQ(6, unit.volumes.size());
    }
}

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, simple_cms)
{
    verbose_ = true;
    std::string const basename = "simple-cms";
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
    write_org_json(result, basename);

    ASSERT_EQ(1, result.universes.size());
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        EXPECT_EQ(8, unit.volumes.size());
        VolumeInstanceAccessor get_vi_id{unit.volumes};
        EXPECT_EQ("[EXTERIOR]@world", get_vi_id(0));
        EXPECT_EQ("1", get_vi_id(1));  // vacuum_tube_pv
        EXPECT_EQ("2", get_vi_id(2));  // si_tracker_pv
        EXPECT_EQ("0", get_vi_id(7));  // world_PV
    }
}

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, testem3)
{
    std::string const basename = "testem3";
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
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
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
    write_org_json(result, basename);

    ASSERT_EQ(1, result.universes.size());
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        EXPECT_EQ(4, unit.volumes.size());
        VolumeInstanceAccessor get_vi_id{unit.volumes};
        // See GeoTests
        EXPECT_EQ("1", get_vi_id(1));  // Tile_Plug1Module
        EXPECT_EQ("2", get_vi_id(2));  // Tile_Absorber
        EXPECT_EQ("0", get_vi_id(3));  // Tile_ITCModule (world volume)
    }
}

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, znenv)
{
    verbose_ = true;
    std::string const basename = "znenv";
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
    write_org_json(result, basename);

    EXPECT_EQ(9, result.universes.size());
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        EXPECT_EQ(6, unit.volumes.size());
        VolumeInstanceAccessor get_vi_id{unit.volumes};
        // World PV label doesn't get repliaced
        EXPECT_EQ(VolumeId{}, unit.background.label);
        // World PV
        EXPECT_EQ("0", get_vi_id(5));
    }
    {
        auto const& unit = std::get<UnitInput>(result.universes[4]);
        VolumeInstanceAccessor get_vi_id{unit.volumes};
        EXPECT_EQ("ZNST", unit.label);
        EXPECT_EQ(VolumeId{8}, unit.background.label);  // ZNST
        ASSERT_LT(unit.background.volume, unit.volumes.size());
        // Implementation volume name
        EXPECT_EQ("[BG]@ZNST", get_vi_id(unit.background.volume));
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

    this->load_gdml(filename);
    auto convert = this->make_converter(filename);
    auto input = convert(this->geo(), *this->volumes()).input;

    write_org_json(input, filename);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
