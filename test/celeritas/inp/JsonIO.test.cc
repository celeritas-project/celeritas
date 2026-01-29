//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/JsonIO.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/inp/ControlIO.json.hh"
#include "celeritas/inp/EventsIO.json.hh"
#include "celeritas/inp/StandaloneInputIO.json.hh"
#include "celeritas/inp/SystemIO.json.hh"
#include "celeritas/inp/TrackingIO.json.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace inp
{
namespace test
{
//---------------------------------------------------------------------------//
//! Verify JSON round-trip serialization.
template<class T>
void verify_json_round_trip(T const& input, char const* expected)
{
    nlohmann::json obj(input);
    EXPECT_JSON_EQ(expected, obj.dump());

    auto rt_input = obj.get<T>();
    EXPECT_JSON_EQ(expected, nlohmann::json(rt_input).dump());
}

//---------------------------------------------------------------------------//
TEST(JsonIOTest, control)
{
    Control input;
    input.capacity = CoreStateCapacity::from_default(false);
    input.optical_capacity = OpticalStateCapacity::from_default(false);
    input.track_order = TrackOrder::init_charge;
    input.seed = 12345;

    char const expected[]
        = R"json({"capacity":{"events":null,"initializers":32768,"primaries":4096,"secondaries":8192,"tracks":4096},"device_debug":null,"optical_capacity":{"generators":8192,"primaries":524288,"tracks":4096},"seed":12345,"track_order":1,"warm_up":false})json";
    verify_json_round_trip(input, expected);
}

TEST(JsonIOTest, events)
{
    {
        // Test optical EM generator round trip
        OpticalGenerator input = OpticalEmGenerator{};

        char const expected[] = R"json({"_type":"em"})json";
        verify_json_round_trip(input, expected);
    }
    {
        using Real3 = Array<double, 3>;

        // Test optical primary generator round trip
        OpticalGenerator input = [] {
            OpticalPrimaryGenerator opg;
            opg.primaries = 512;
            opg.energy = NormalDistribution{1, 0};
            opg.angle = MonodirectionalDistribution{{0, 0, 1}};
            opg.shape = UniformBoxDistribution{{0, 0, 0}, {1, 1, 1}};
            return opg;
        }();

        char const expected[]
            = R"json({"_type":"primary","angle":{"_type":"delta","value":[0.0,0.0,1.0]},"energy":{"_type":"normal","mean":1.0,"stddev":0.0},"primaries":512,"shape":{"_type":"uniform_box","lower":[0.0,0.0,0.0],"upper":[1.0,1.0,1.0]}})json";
        verify_json_round_trip(input, expected);

        auto rt_input = nlohmann::json(input).get<OpticalPrimaryGenerator>();
        EXPECT_EQ(512, rt_input.primaries);

        auto const& energy = std::get<NormalDistribution>(rt_input.energy);
        EXPECT_EQ(1.0, energy.mean);
        EXPECT_EQ(0.0, energy.stddev);

        auto const& angle
            = std::get<MonodirectionalDistribution>(rt_input.angle);
        EXPECT_EQ(Real3({0, 0, 1}), angle.value);

        auto const& shape = std::get<UniformBoxDistribution>(rt_input.shape);
        EXPECT_EQ(Real3({0, 0, 0}), shape.lower);
        EXPECT_EQ(Real3({1, 1, 1}), shape.upper);
    }
}

TEST(JsonIOTest, standalone_input)
{
    OpticalStandaloneInput input;
    input.problem.model.geometry = "geometry.gdml";
    input.problem.capacity = OpticalStateCapacity::from_default(false);
    input.problem.limits.steps = 1000;
    input.problem.limits.step_iters = 10000;

    char const expected[]
        = R"json({"_format":"optical-standalone-input","_version":"0.7.0","geant_setup":{"_format":"geant4-optical-physics","_version":"0.7.0","absorption":true,"boundary":{"enable":true,"invoke_sd":false},"cherenkov":{"enable":true,"max_beta_change":10.0,"max_photons":100,"stack_photons":true,"track_secondaries_first":true},"mie_scattering":true,"rayleigh_scattering":true,"scintillation":{"by_particle_type":false,"enable":true,"finite_rise_time":false,"stack_photons":true,"track_info":false,"track_secondaries_first":true},"verbose":false,"wavelength_shifting":{"time_profile":"delta"},"wavelength_shifting2":{"time_profile":"delta"}},"problem":{"capacity":{"generators":8192,"primaries":524288,"tracks":4096},"generator":{"_type":"em"},"limits":{"step_iters":10000,"steps":1000},"model":{"geometry":"geometry.gdml"},"output_file":"-","perfetto_file":null,"seed":0,"timers":{"action":false,"step":false}},"system":{"device":null,"environment":{}}})json";
    verify_json_round_trip(input, expected);
}

TEST(JsonIOTest, system)
{
    System input;
    input.environment = {{"TWO", "2"}, {"ONE", "1"}};
    {
        // Without optional device
        char const expected[]
            = R"json({"device":null,"environment":{"ONE":"1","TWO":"2"}})json";
        verify_json_round_trip(input, expected);
    }

    input.device.emplace();
    input.device->stack_size = 1024;
    input.device->heap_size = 8192;
    {
        // With device
        char const expected[]
            = R"json({"device":{"heap_size":8192,"stack_size":1024},"environment":{"ONE":"1","TWO":"2"}})json";
        verify_json_round_trip(input, expected);
    }
}

TEST(JsonIOTest, tracking)
{
    Tracking input;
    input.limits.steps = 1000;
    input.limits.step_iters = 10000;
    input.limits.field_substeps = 100;
    input.optical_limits.steps = 0;
    input.optical_limits.step_iters = 0;

    char const expected[]
        = R"json({"force_step_limit":0.0,"limits":{"field_substeps":100,"step_iters":10000,"steps":1000},"optical_limits":{"step_iters":0,"steps":0}})json";
    verify_json_round_trip(input, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace inp
}  // namespace celeritas
