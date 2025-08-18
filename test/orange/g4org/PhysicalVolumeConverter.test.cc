//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/PhysicalVolumeConverter.test.cc
//---------------------------------------------------------------------------//
#include "orange/g4org/PhysicalVolumeConverter.hh"

#include "corecel/io/Logger.hh"
#include "corecel/io/StreamableVariant.hh"
#include "corecel/sys/Environment.hh"
#include "geocel/GeantGeoParams.hh"
#include "orange/MatrixUtils.hh"
#include "orange/orangeinp/ObjectInterface.hh"
#include "orange/transform/TransformIO.hh"

#include "GeantLoadTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace g4org
{
namespace test
{
//---------------------------------------------------------------------------//
constexpr RealTurn degrees_to_turn(double v)
{
    return RealTurn{static_cast<real_type>(v / 360)};
}

auto make_options()
{
    PhysicalVolumeConverter::Options opts;
    opts.verbose = false;
    opts.scale = 0.1;
    return opts;
}

//---------------------------------------------------------------------------//
class PhysicalVolumeConverterTest : public GeantLoadTestBase
{
  public:
    Label const& get_label(LogicalVolume const& lv)
    {
        CELER_EXPECT(lv.id);
        return this->geo().impl_volumes().at(lv.id);
    }

    Label const& get_label(PhysicalVolume const& pv)
    {
        CELER_EXPECT(pv.id);
        return this->geo().volume_instances().at(pv.id);
    }

    G4VPhysicalVolume const& world() const { return *this->geo().world(); }
};

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, four_levels)
{
    this->load_test_gdml("four-levels");
    PhysicalVolumeConverter::Options opts;
    opts.verbose = false;
    opts.scale = 0.1;
    PhysicalVolumeConverter convert{this->geo(), make_options()};

    PhysicalVolume world = convert(this->world());
    EXPECT_EQ("World_PV", this->get_label(world).name);
    if (!std::holds_alternative<NoTransformation>(world.transform))
    {
        ADD_FAILURE() << "Unexpected transform type: "
                      << StreamableVariant{world.transform};
    }

    ASSERT_TRUE(world.lv);
    EXPECT_EQ(1, world.lv.use_count());
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, intersection_boxes)
{
    this->load_test_gdml("intersection-boxes");

    PhysicalVolumeConverter convert{this->geo(), make_options()};
    PhysicalVolume world = convert(this->world());

    ASSERT_TRUE(world.lv);
    EXPECT_EQ(Label{"world"}, this->get_label(*world.lv));
    ASSERT_EQ(1, world.lv->children.size());

    auto const& inner_pv = world.lv->children.front();
    ASSERT_TRUE(inner_pv.lv);
    EXPECT_EQ(Label{"inner"}, this->get_label(*inner_pv.lv));
    ASSERT_TRUE(inner_pv.lv->solid);
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

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, DISABLED_solids)
{
    celeritas::environment().insert({"G4ORG_ALLOW_ERRORS", "1"});
    this->load_test_gdml("solids");

    PhysicalVolumeConverter convert{this->geo(), make_options()};

    PhysicalVolume world = convert(this->world());
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, testem3)
{
    this->load_test_gdml("testem3");
    PhysicalVolumeConverter convert{this->geo(), make_options()};

    PhysicalVolume world = convert(this->world());
    EXPECT_EQ("world_PV", this->get_label(world).name);

    ASSERT_TRUE(world.lv);
    EXPECT_EQ(1, world.lv.use_count());
    LogicalVolume const* lv = world.lv.get();

    {
        // Test world's logical volume
        EXPECT_EQ(Label{"world"}, this->get_label(*lv));
        ASSERT_TRUE(lv->solid);
        EXPECT_JSON_EQ(
            R"json({"_type":"shape","interior":{"_type":"box","halfwidths":[24.0,24.0,24.0]},"label":"World"})json",
            to_string(*lv->solid));
        ASSERT_EQ(1, lv->children.size());

        auto const& calo_pv = lv->children.front();
        EXPECT_EQ(1, calo_pv.lv.use_count());
        ASSERT_TRUE(calo_pv.lv);
        lv = calo_pv.lv.get();
    }
    {
        // Test calorimeter
        EXPECT_EQ(Label{"calorimeter"}, this->get_label(*lv));
        ASSERT_EQ(50, lv->children.size());

        auto const& first_layer = lv->children.front();
        EXPECT_EQ(50, first_layer.lv.use_count());
        if (auto* trans = std::get_if<Translation>(&first_layer.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{-19.6, 0, 0}), trans->translation());
        }
        else
        {
            ADD_FAILURE() << "Unexpected transform type: "
                          << StreamableVariant{first_layer.transform};
        }

        auto const& last_layer = lv->children.back();
        EXPECT_EQ(first_layer.lv.get(), last_layer.lv.get());

        ASSERT_TRUE(first_layer.lv);
        lv = first_layer.lv.get();
    }
    {
        // Test layer
        EXPECT_EQ(Label{"layer"}, this->get_label(*lv));
        ASSERT_EQ(2, lv->children.size());

        ASSERT_TRUE(lv->solid);
        EXPECT_JSON_EQ(
            R"json({"_type":"shape","interior":{"_type":"box","halfwidths":[0.4,20.0,20.0]},"label":"Layer"})json",
            to_string(*lv->solid));

        auto const& lead = lv->children.front();
        EXPECT_EQ(1, lead.lv.use_count());

        ASSERT_TRUE(lead.lv);
        lv = lead.lv.get();
    }
    {
        // Test lead
        EXPECT_EQ(Label{"pb"}, this->get_label(*lv));
        EXPECT_EQ(0, lv->children.size());
    }
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, transformed_box)
{
    this->load_test_gdml("transformed-box");

    PhysicalVolumeConverter convert{this->geo(), make_options()};
    PhysicalVolume world = convert(this->world());
    EXPECT_EQ(Label{"world_PV"}, this->get_label(world));

    ASSERT_TRUE(world.lv);
    ASSERT_EQ(3, world.lv->children.size());

    {
        auto const& pv = world.lv->children[0];
        EXPECT_EQ("transrot", this->get_label(pv).name);
        if (auto* trans = std::get_if<Transformation>(&pv.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{0, 0, -10}), trans->translation());
            auto mat = make_rotation(Axis::y, degrees_to_turn(30));
            mat = make_transpose(mat);
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
        auto const& pv = world.lv->children[1];
        EXPECT_EQ("default", this->get_label(pv).name);
        if (!std::holds_alternative<NoTransformation>(pv.transform))
        {
            ADD_FAILURE() << "Unexpected transform type: "
                          << StreamableVariant{pv.transform};
        }
    }
    {
        auto const& pv = world.lv->children[2];
        EXPECT_EQ("trans", this->get_label(pv).name);
        if (auto* trans = std::get_if<Translation>(&pv.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{0, 0, 10}), trans->translation());
        }
        else
        {
            ADD_FAILURE() << "Unexpected transform type: "
                          << StreamableVariant{pv.transform};
        }
    }
    {
        auto const& lv_parent = world.lv->children[1].lv;
        CELER_ASSERT(lv_parent);
        ASSERT_EQ(1, lv_parent->children.size());
        auto const& pv = lv_parent->children[0];
        EXPECT_EQ("rot", this->get_label(pv).name);
        if (auto* trans = std::get_if<Transformation>(&pv.transform))
        {
            EXPECT_VEC_SOFT_EQ((Real3{0, 0, 0}), trans->translation());
            auto mat = make_rotation(Axis::x, degrees_to_turn(90));
            mat = make_rotation(Axis::y, degrees_to_turn(-87.1875), mat);
            mat = make_rotation(Axis::z, degrees_to_turn(90), mat);
            mat = make_transpose(mat);
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
}

//---------------------------------------------------------------------------//
TEST_F(PhysicalVolumeConverterTest, znenv)
{
    this->load_test_gdml("znenv");
    PhysicalVolumeConverter convert{this->geo(), make_options()};
    PhysicalVolume world = convert(this->world());
    (void)sizeof(world);

    // TODO: test parameterisation ReplicaId
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace g4org
}  // namespace celeritas
