//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/JsonIO.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/inp/EventsIO.json.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace inp
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(JsonIOTest, events)
{
    using Real3 = Array<double, 3>;

    // Test optical primary generator round trip
    OpticalPrimaryGenerator input;
    input.primaries = 512;
    input.energy = NormalDistribution{1, 0};
    input.angle = MonodirectionalDistribution{{0, 0, 1}};
    input.shape = UniformBoxDistribution{{0, 0, 0}, {1, 1, 1}};

    char const expected[]
        = R"json({"angle":{"_type":"delta","value":[0.0,0.0,1.0]},"energy":{"_type":"normal","mean":1.0,"stddev":0.0},"primaries":512,"shape":{"_type":"uniform_box","lower":[0.0,0.0,0.0],"upper":[1.0,1.0,1.0]}})json";
    nlohmann::json obj(input);
    EXPECT_JSON_EQ(expected, obj.dump());

    auto rt_input = obj.get<OpticalPrimaryGenerator>();
    EXPECT_JSON_EQ(expected, nlohmann::json(rt_input).dump());

    EXPECT_EQ(input.primaries, rt_input.primaries);

    auto const& energy = std::get<NormalDistribution>(rt_input.energy);
    EXPECT_EQ(1.0, energy.mean);
    EXPECT_EQ(0.0, energy.stddev);

    auto const& angle = std::get<MonodirectionalDistribution>(rt_input.angle);
    EXPECT_EQ(Real3({0, 0, 1}), angle.value);

    auto const& shape = std::get<UniformBoxDistribution>(rt_input.shape);
    EXPECT_EQ(Real3({0, 0, 0}), shape.lower);
    EXPECT_EQ(Real3({1, 1, 1}), shape.upper);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace inp
}  // namespace celeritas
