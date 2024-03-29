//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/PhysicalVolumeConverter.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/PhysicalVolumeConverter.hh"

#include <regex>

#include "corecel/io/StreamableVariant.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/LazyGeoManager.hh"
#include "geocel/UnitUtils.hh"
#include "orange/orangeinp/ObjectInterface.hh"
#include "orange/transform/TransformIO.hh"

#include "celeritas_test.hh"

using celeritas::test::to_cm;

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
std::string simplify_pointers(std::string const& s)
{
    static std::regex const subs_ptr("0x[0-9a-f]+");
    return std::regex_replace(s, subs_ptr, "0x0");
}

class PhysicalVolumeConverterTest : public ::celeritas::test::Test
{
  protected:
    G4VPhysicalVolume const* load(std::string const& filename)
    {
        return ::celeritas::load_geant_geometry_native(
            this->test_data_path("geocel", filename));
    }

    void TearDown() final { ::celeritas::reset_geant_geometry(); }
};

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, testem3)
{
    G4VPhysicalVolume const* g4world = this->load("testem3.gdml");
    PhysicalVolumeConverter convert{/* verbose = */ true};

    PhysicalVolume world = convert(*g4world);
    EXPECT_EQ("World_PV", world.name);
    EXPECT_EQ(0, world.copy_number);

    ASSERT_TRUE(world.lv);
    EXPECT_EQ(1, world.lv.use_count());
    LogicalVolume const* lv = world.lv.get();

    {
        // Test world's logical volume
        EXPECT_NE(nullptr, lv->g4lv);
        EXPECT_EQ("World0x0", simplify_pointers(lv->name));
        ASSERT_TRUE(lv->solid);
        if (CELERITAS_USE_JSON)
        {
            EXPECT_JSON_EQ(
                R"json({"_type":"shape","interior":{"_type":"box","halfwidths":[24.0,24.0,24.0]},"label":"World"})json",
                to_string(*lv->solid));
        }
        ASSERT_EQ(1, lv->daughters.size());

        auto const& calo_pv = lv->daughters.front();
        EXPECT_EQ(1, calo_pv.lv.use_count());
        ASSERT_TRUE(calo_pv.lv);
        lv = calo_pv.lv.get();
    }
    {
        // Test calorimeter
        EXPECT_EQ("Calorimeter0x0", simplify_pointers(lv->name));
        ASSERT_EQ(50, lv->daughters.size());

        auto const& first_layer = lv->daughters.front();
        EXPECT_EQ(1, first_layer.copy_number);
        EXPECT_EQ(50, first_layer.lv.use_count());
        if (auto* trans = std::get_if<Translation>(&first_layer.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{-19.6, 0, 0}),
                               to_cm(trans->translation()));
        }
        else
        {
            ADD_FAILURE() << "Unexpected transform type: "
                          << StreamableVariant{first_layer.transform};
        }

        auto const& last_layer = lv->daughters.back();
        EXPECT_EQ(50, last_layer.copy_number);
        EXPECT_EQ(first_layer.lv.get(), last_layer.lv.get());

        ASSERT_TRUE(first_layer.lv);
        lv = first_layer.lv.get();
    }
    {
        // Test layer
        EXPECT_EQ("Layer0x0", simplify_pointers(lv->name));
        ASSERT_EQ(2, lv->daughters.size());

        ASSERT_TRUE(lv->solid);
        if (CELERITAS_USE_JSON)
        {
            EXPECT_JSON_EQ(
                R"json({"_type":"shape","interior":{"_type":"box","halfwidths":[0.4,20.0,20.0]},"label":"Layer"})json",
                to_string(*lv->solid));
        }

        auto const& lead = lv->daughters.front();
        EXPECT_EQ(1, lead.lv.use_count());

        ASSERT_TRUE(lead.lv);
        lv = lead.lv.get();
    }
    {
        // Test lead
        EXPECT_EQ("G4_Pb0x0", simplify_pointers(lv->name));
        EXPECT_EQ(0, lv->daughters.size());
    }
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, DISABLED_four_levels)
{
    G4VPhysicalVolume const* g4world = this->load("four-levels.gdml");
    PhysicalVolumeConverter convert{/* verbose = */ true};

    PhysicalVolume world = convert(*g4world);
    EXPECT_EQ("World_PV", world.name);
    EXPECT_EQ(0, world.copy_number);

    ASSERT_TRUE(world.lv);
    EXPECT_EQ(1, world.lv.use_count());
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, znenv)
{
    G4VPhysicalVolume const* g4world = this->load("znenv.gdml");

    PhysicalVolumeConverter convert{/* verbose = */ true};

    PhysicalVolume world = convert(*g4world);
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, DISABLED_solids)
{
    G4VPhysicalVolume const* g4world = this->load("solids.gdml");

    PhysicalVolumeConverter convert{/* verbose = */ true};

    PhysicalVolume world = convert(*g4world);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
