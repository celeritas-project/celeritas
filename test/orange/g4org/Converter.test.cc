//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Converter.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/Converter.hh"

#include <fstream>
#include <variant>

#include "corecel/OpaqueIdIO.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
#include "geocel/Types.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"
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
    using VecStr = std::vector<std::string>;
    void SetUp() override { verbose_ = !celeritas::getenv("VERBOSE").empty(); }

    //! Make a converter
    Converter make_converter(std::string_view filename = {})
    {
        inp::OrangeGeoFromGeant opts;
        opts.verbose_structure = verbose_;
        if (!filename.empty())
        {
            opts.objects_output_file = std::string(filename) + ".objects.json";
            opts.csg_output_file = std::string(filename) + ".csg.json";
        }
        return Converter{std::move(opts)};
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

    VecStr local_parent_map(UnitInput const& u) const;

    bool verbose_{false};
};

class VolumeInstanceAccessor
{
  public:
    using VolumesVector = std::vector<VolumeInput>;

    explicit VolumeInstanceAccessor(VolumesVector const& volumes)
        : volumes_(volumes), params_(nullptr)
    {
    }

    VolumeInstanceAccessor(VolumesVector const& volumes,
                           VolumeParams const& params)
        : volumes_(volumes), params_(&params)
    {
    }

    std::string operator()(LocalVolumeId lv_id) const;
    std::string operator()(size_type i) const;

  private:
    VolumesVector const& volumes_;
    VolumeParams const* params_;
};

//---------------------------------------------------------------------------//

auto ConverterTest::local_parent_map(UnitInput const& u) const -> VecStr
{
    CELER_ASSERT(this->volumes());
    VolumeInstanceAccessor get_label{u.volumes, *this->volumes()};
    VecStr result;
    for (auto const& [src, tgt] : u.local_parent_map)
    {
        std::ostringstream os;
        os << get_label(src) << "->" << get_label(tgt);
        result.emplace_back(std::move(os).str());
    }
    return result;
}

//---------------------------------------------------------------------------//

std::string VolumeInstanceAccessor::operator()(LocalVolumeId lv_id) const
{
    if (!lv_id)
        return "<null lv>";

    return (*this)(lv_id.get());
}

std::string VolumeInstanceAccessor::operator()(size_type i) const
{
    if (i >= volumes_.size())
        return "<out of bounds>";

    auto const& var_label = volumes_[i].label;
    if (auto* label = std::get_if<Label>(&var_label))
        return to_string(*label);

    CELER_ASSUME(std::holds_alternative<VolumeInstanceId>(var_label));
    auto vi_id = std::get<VolumeInstanceId>(var_label);
    if (!vi_id)
        return "<null vi>";

    // Volume params available: return PV label
    if (params_)
        return to_string(params_->volume_instance_labels().at(vi_id));

    // Return ID value
    return std::to_string(vi_id.get());
}

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
    std::string const basename = "simple-cms";
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
    write_org_json(result, basename);

    ASSERT_EQ(1, result.universes.size());
    ASSERT_TRUE(std::holds_alternative<UnitInput>(result.universes[0]));
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        EXPECT_EQ(8, unit.volumes.size());
        VolumeInstanceAccessor get_vi_id{unit.volumes};
        EXPECT_EQ("[EXTERIOR]@world", get_vi_id(0));
        EXPECT_EQ("1", get_vi_id(1));  // vacuum_tube_pv
        EXPECT_EQ("2", get_vi_id(2));  // si_tracker_pv
        EXPECT_EQ("0", get_vi_id(7));  // world_PV

        static char const* const expected_local_parent_map[] = {
            "vacuum_tube_pv->world_PV",
            "si_tracker_pv->world_PV",
            "em_calorimeter_pv->world_PV",
            "had_calorimeter_pv->world_PV",
            "sc_solenoid_pv->world_PV",
            "iron_muon_chambers_pv->world_PV",
        };
        EXPECT_VEC_EQ(expected_local_parent_map, this->local_parent_map(unit));
    }
}

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, multilevel)
{
    std::string const basename = "multi-level";
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
    write_org_json(result, basename);

    ASSERT_EQ(3, result.universes.size());
    ASSERT_TRUE(std::holds_alternative<UnitInput>(result.universes[0]));
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        SCOPED_TRACE("universe 0");
        EXPECT_EQ(Label{"world"}, unit.label);
        EXPECT_EQ(7, unit.volumes.size());
        EXPECT_EQ(17, unit.surfaces.size());

        static char const* const expected_local_parent_map[] = {
            "topbox1->world_PV",
            "topbox2->world_PV",
            "topbox3->world_PV",
            "topbox4->world_PV",
            "topsph1->world_PV",
        };
        EXPECT_VEC_EQ(expected_local_parent_map, this->local_parent_map(unit));
    }
    ASSERT_TRUE(std::holds_alternative<UnitInput>(result.universes[1]));
    {
        auto const& unit = std::get<UnitInput>(result.universes[1]);
        SCOPED_TRACE("universe 1");
        EXPECT_EQ(Label{"box"}, unit.label);
        EXPECT_EQ(5, unit.volumes.size());
        EXPECT_EQ(7, unit.surfaces.size());

        static char const* const expected_local_parent_map[] = {
            "boxsph1@0->[BG]@box",
            "boxsph2@0->[BG]@box",
            "boxtri@0->[BG]@box",
        };
        EXPECT_VEC_EQ(expected_local_parent_map, this->local_parent_map(unit));
    }
    ASSERT_TRUE(std::holds_alternative<UnitInput>(result.universes[2]));
    {
        auto const& unit = std::get<UnitInput>(result.universes[2]);
        SCOPED_TRACE("universe 2");
        EXPECT_EQ(Label{"box_refl"}, unit.label);
        EXPECT_EQ(5, unit.volumes.size());
        EXPECT_EQ(7, unit.surfaces.size());

        static char const* const expected_local_parent_map[] = {
            "boxsph1@1->[BG]@box_refl",
            "boxsph2@1->[BG]@box_refl",
            "boxtri@1->[BG]@box_refl",
        };
        EXPECT_VEC_EQ(expected_local_parent_map, this->local_parent_map(unit));
    }
}

//---------------------------------------------------------------------------//
TEST_F(ConverterTest, testem3)
{
    std::string const basename = "testem3";
    verbose_ = true;
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

        static char const* const expected_local_parent_map[] = {
            "Layer@0->Calorimeter",  "Layer@1->Calorimeter",
            "Layer@2->Calorimeter",  "Layer@3->Calorimeter",
            "Layer@4->Calorimeter",  "Layer@5->Calorimeter",
            "Layer@6->Calorimeter",  "Layer@7->Calorimeter",
            "Layer@8->Calorimeter",  "Layer@9->Calorimeter",
            "Layer@10->Calorimeter", "Layer@11->Calorimeter",
            "Layer@12->Calorimeter", "Layer@13->Calorimeter",
            "Layer@14->Calorimeter", "Layer@15->Calorimeter",
            "Layer@16->Calorimeter", "Layer@17->Calorimeter",
            "Layer@18->Calorimeter", "Layer@19->Calorimeter",
            "Layer@20->Calorimeter", "Layer@21->Calorimeter",
            "Layer@22->Calorimeter", "Layer@23->Calorimeter",
            "Layer@24->Calorimeter", "Layer@25->Calorimeter",
            "Layer@26->Calorimeter", "Layer@27->Calorimeter",
            "Layer@28->Calorimeter", "Layer@29->Calorimeter",
            "Layer@30->Calorimeter", "Layer@31->Calorimeter",
            "Layer@32->Calorimeter", "Layer@33->Calorimeter",
            "Layer@34->Calorimeter", "Layer@35->Calorimeter",
            "Layer@36->Calorimeter", "Layer@37->Calorimeter",
            "Layer@38->Calorimeter", "Layer@39->Calorimeter",
            "Layer@40->Calorimeter", "Layer@41->Calorimeter",
            "Layer@42->Calorimeter", "Layer@43->Calorimeter",
            "Layer@44->Calorimeter", "Layer@45->Calorimeter",
            "Layer@46->Calorimeter", "Layer@47->Calorimeter",
            "Layer@48->Calorimeter", "Layer@49->Calorimeter",
            "Calorimeter->world_PV",
        };
        EXPECT_VEC_EQ(expected_local_parent_map, this->local_parent_map(*unit));
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

        static char const* const expected_local_parent_map[]
            = {"pb_pv->[BG]@layer", "lar_pv->[BG]@layer"};
        EXPECT_VEC_EQ(expected_local_parent_map, this->local_parent_map(*unit));
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
    std::string const basename = "znenv";
    this->load_test_gdml(basename);
    auto convert = this->make_converter(basename);
    auto result = convert(this->geo(), *this->volumes()).input;
    write_org_json(result, basename);

    ASSERT_EQ(9, result.universes.size());
    {
        auto const& unit = std::get<UnitInput>(result.universes[0]);
        EXPECT_EQ(6, unit.volumes.size());
        VolumeInstanceAccessor get_vi_id{unit.volumes};
        // World PV label doesn't get replicated
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
