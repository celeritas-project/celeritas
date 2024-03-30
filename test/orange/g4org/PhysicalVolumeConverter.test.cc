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
#include "corecel/sys/Environment.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/LazyGeoManager.hh"
#include "geocel/UnitUtils.hh"
#include "orange/MatrixUtils.hh"
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
TEST_F(PhysicalVolumeConverterTest, intersection_boxes)
{
    G4VPhysicalVolume const* g4world = this->load("intersection-boxes.gdml");

    PhysicalVolumeConverter convert{/* verbose = */ true};
    PhysicalVolume world = convert(*g4world);

    ASSERT_TRUE(world.lv);
    EXPECT_EQ("world0x0", simplify_pointers(world.lv->name));
    ASSERT_EQ(1, world.lv->daughters.size());

    auto const& inner_pv = world.lv->daughters.front();
    ASSERT_TRUE(inner_pv.lv);
    EXPECT_EQ("inner0x0", simplify_pointers(inner_pv.lv->name));
    ASSERT_TRUE(inner_pv.lv->solid);
    if (CELERITAS_USE_JSON)
    {
        EXPECT_JSON_EQ(
            R"json(
{"_type":"all","daughters":[
  {"_type":"shape","interior": {"_type":"box","halfwidths":[1.0,1.5,2.0]},"label":"first"},
  {"_type":"transformed",
   "daughter": {"_type":"shape","interior": {"_type":"box","halfwidths":[1.5,2.0,2.5]},"label":"second"},
   "transform":{"_type":"transformation", "data":
[0.8660254037844388,0.0,0.5,
 0.0,1.0,0.0,
 -0.5,0.0,0.8660254037844388,
 1.0,2.0,4.0]}}],"label":"isect"})json",
            to_string(*inner_pv.lv->solid));
    }
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, DISABLED_solids)
{
    celeritas::environment().insert({"G4ORG_ALLOW_ERRORS", "1"});
    G4VPhysicalVolume const* g4world = this->load("solids.gdml");

    PhysicalVolumeConverter convert{/* verbose = */ true};

    PhysicalVolume world = convert(*g4world);
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, testem3)
{
    G4VPhysicalVolume const* g4world = this->load("testem3.gdml");
    PhysicalVolumeConverter convert{/* verbose = */ false};

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
TEST_F(PhysicalVolumeConverterTest, transformed_box)
{
    G4VPhysicalVolume const* g4world = this->load("transformed-box.gdml");

    PhysicalVolumeConverter convert{/* verbose = */ false};
    PhysicalVolume world = convert(*g4world);
    EXPECT_EQ("world_PV", simplify_pointers(world.name));

    ASSERT_TRUE(world.lv);
    ASSERT_EQ(3, world.lv->daughters.size());

    {
        auto const& pv = world.lv->daughters[0];
        EXPECT_EQ("transrot", pv.name);
        if (auto* trans = std::get_if<Transformation>(&pv.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{0, 0, -10}), to_cm(trans->translation()));
            auto const mat = make_rotation(Axis::y, Turn{30.0 / 360.0});
            EXPECT_VEC_SOFT_EQ(mat[0], trans->rotation()[0]);
            EXPECT_VEC_SOFT_EQ(mat[1], trans->rotation()[1]);
            EXPECT_VEC_SOFT_EQ(mat[2], trans->rotation()[2]);
        }
        else
        {
            ADD_FAILURE() << "Unexpected transform type: "
                          << StreamableVariant{pv.transform};
        }
    }
    {
        auto const& pv = world.lv->daughters[1];
        EXPECT_EQ("default", pv.name);
        if (auto* trans = std::get_if<Translation>(&pv.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), to_cm(trans->translation()));
        }
        else
        {
            ADD_FAILURE() << "Unexpected transform type: "
                          << StreamableVariant{pv.transform};
        }
    }
    {
        auto const& pv = world.lv->daughters[2];
        EXPECT_EQ("trans", pv.name);
        if (auto* trans = std::get_if<Translation>(&pv.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{0, 0, 10}), to_cm(trans->translation()));
        }
        else
        {
            ADD_FAILURE() << "Unexpected transform type: "
                          << StreamableVariant{pv.transform};
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
