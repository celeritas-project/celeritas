//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/JsonIO.test.cc
//---------------------------------------------------------------------------//
// TODO: combine all inp/IO.json
#include "corecel/inp/DistributionsIO.json.hh"
#include "celeritas/inp/ControlIO.json.hh"
#include "celeritas/inp/DiagnosticsIO.json.hh"
#include "celeritas/inp/EventsIO.json.hh"
#include "celeritas/inp/FieldIO.json.hh"
#include "celeritas/inp/ImportIO.json.hh"
#include "celeritas/inp/ScoringIO.json.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/inp/StandaloneInputIO.json.hh"
#include "celeritas/inp/SystemIO.json.hh"
#include "celeritas/inp/TrackingIO.json.hh"

#include "JsonTestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace inp
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(JsonIO, control)
{
    Control input;
    input.capacity = [] {
        inp::CoreStateCapacity cap;
        cap.tracks = 4096;
        cap.primaries = *cap.tracks;
        cap.initializers = 8 * *cap.tracks;
        cap.secondaries = 2 * *cap.tracks;
        return cap;
    }();
    input.optical_capacity = [] {
        inp::OpticalStateCapacity cap;
        cap.tracks = 4096;
        cap.primaries = 128 * *cap.tracks;
        cap.generators = 2 * *cap.tracks;
        return cap;
    }();
    input.track_order = TrackOrder::init_charge;
    input.seed = 12345;

    static char const expected[]
        = R"json({"capacity":{"events":null,"initializers":32768,"primaries":4096,"secondaries":8192,"tracks":4096},"device_debug":null,"optical_capacity":{"generators":8192,"primaries":524288,"tracks":4096},"seed":12345,"track_order":"init_charge","warm_up":false})json";
    EXPECT_JSON_ROUND_TRIP(input, expected);
}

TEST(JsonIO, diagnostics)
{
    Diagnostics input;
    input.export_files.physics = "physics.root";
    input.export_files.offload = "offload.jsonl";
    input.export_files.geometry = "geometry.gdml";
    input.slot = [] {
        SlotDiagnostic sd;
        sd.basename = "slot";
        return sd;
    }();
    input.mctruth = [] {
        McTruth mct;
        mct.output_file = "mctruth.root";
        return mct;
    }();
    input.step.emplace();

    static char const expected[]
        = R"json({"action":false,"counters":{"event":true,"step":true},"export_files":{"geometry":"geometry.gdml","offload":"offload.jsonl","physics":"physics.root"},"log_frequency":1,"mctruth":{"filter":{"event_id":null,"parent_id":null,"post_step_action_id":null,"track_id":[]},"output_file":"mctruth.root"},"output_file":"-","perfetto_file":"","slot":{"basename":"slot"},"status_checker":false,"step":{"bins":1000},"timers":{"action":false,"step":false}})json";
    EXPECT_JSON_ROUND_TRIP(input, expected);
}

TEST(JsonIO, events)
{
    {
        // Test optical EM generator round trip
        OpticalGenerator input = OpticalEmGenerator{};

        static char const expected[] = R"json({"_type":"em"})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
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

        static char const expected[]
            = R"json({"_type":"primary","angle":{"_type":"delta","value":[0.0,0.0,1.0]},"energy":{"_type":"normal","mean":1.0,"stddev":0.0},"primaries":512,"shape":{"_type":"uniform_box","lower":[0.0,0.0,0.0],"upper":[1.0,1.0,1.0]}})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);

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
    {
        TruncatedDistribution<NormalDistribution> input;
        input.distribution.mean = 2.0;
        input.distribution.stddev = 1.0;
        input.lower = 0.0;

        static char const expected[]
            = R"json({"_type":"truncated","distribution":{"_type":"normal","mean":2.0,"stddev":1.0},"lower":0.0,"upper":null})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
    {
        Events input;
        input.generator = [] {
            SampleFileEvents sfe;
            sfe.num_events = 8;
            sfe.num_merged = 2;
            sfe.event_file = "events.root";
            sfe.seed = 12345;
            return sfe;
        }();

        static char const expected[]
            = R"json({"generator":{"_type":"sample","event_file":"events.root","num_events":8,"num_merged":2,"seed":12345},"merge":false})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
    {
        Events input;
        input.merge = true;
        input.generator = [] {
            ReadFileEvents rfe;
            rfe.event_file = "events.root";
            return rfe;
        }();

        static char const expected[]
            = R"json({"generator":{"_type":"read","event_file":"events.root"},"merge":true})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
    {
        Events input;
        input.generator = [] {
            CorePrimaryGenerator cpg;
            cpg.energy = NormalDistribution{1, 0};
            cpg.angle = MonodirectionalDistribution{{0, 0, 1}};
            cpg.shape = UniformBoxDistribution{{0, 0, 0}, {1, 1, 1}};
            cpg.num_events = 4;
            cpg.primaries_per_event = 1023;
            cpg.seed = 12345;
            cpg.pdg = {pdg::electron()};
            return cpg;
        }();

        static char const expected[]
            = R"json({"generator":{"_type":"primary","angle":{"_type":"delta","value":[0.0,0.0,1.0]},"energy":{"_type":"normal","mean":1.0,"stddev":0.0},"num_events":4,"pdg":[11],"primaries_per_event":1023,"seed":12345,"shape":{"_type":"uniform_box","lower":[0.0,0.0,0.0],"upper":[1.0,1.0,1.0]}},"merge":false})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
}

TEST(JsonIO, field)
{
    {
        Field input = NoField{};

        static char const expected[] = R"json({"_type":"none"})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
    {
        Field input = [] {
            UniformField field;
            field.strength = {0, 0, 1};
            return field;
        }();

        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        {
            static char const expected[]
                = R"json({"_type":"uniform","driver_options":{"_format":"field-driver","_units":"cgs","_version":"0.7.0","delta_chord":0.025,"delta_intersection":1e-05,"epsilon_rel_max":0.001,"epsilon_step":1e-05,"errcon":0.0001,"max_nsteps":100,"max_stepping_decrease":0.1,"max_stepping_increase":5.0,"max_substeps":10,"minimum_step":1.0000000000000002e-06,"pgrow":-0.2,"pshrink":-0.25,"safety":0.9},"strength":[0.0,0.0,1.0],"units":2})json";
            EXPECT_JSON_ROUND_TRIP(input, expected);
        }
    }
    {
        Field input = [] {
            CylMapField field;
            field.grid_r = {0, 1};
            field.grid_phi = {CylMapField::Turn{0}, CylMapField::Turn{1}};
            field.grid_z = {-100, 100};
            field.field = {0, 1, 2, 3, 4, 5, 6, 7};
            return field;
        }();

        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        {
            static char const expected[]
                = R"json({"_type":"cylmap","driver_options":{"_format":"field-driver","_units":"cgs","_version":"0.7.0","delta_chord":0.025,"delta_intersection":1e-05,"epsilon_rel_max":0.001,"epsilon_step":1e-05,"errcon":0.0001,"max_nsteps":100,"max_stepping_decrease":0.1,"max_stepping_increase":5.0,"max_substeps":10,"minimum_step":1.0000000000000002e-06,"pgrow":-0.2,"pshrink":-0.25,"safety":0.9},"field":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],"grid_phi":[[0.0,"tr"],[1.0,"tr"]],"grid_r":[0.0,1.0],"grid_z":[-100.0,100.0]})json";
            EXPECT_JSON_ROUND_TRIP(input, expected);
        }
    }
    {
        Field input = [] {
            AxisGrid<double> grid;
            grid.min = -10;
            grid.max = 10;
            grid.num = 2;

            CartMapField field;
            field.x = grid;
            field.y = grid;
            field.z = grid;
            field.field = {0, 1, 2, 3, 4, 5, 6, 7};
            return field;
        }();

        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        {
            static char const expected[]
                = R"json({"_type":"cartmap","driver_options":{"_format":"field-driver","_units":"cgs","_version":"0.7.0","delta_chord":0.025,"delta_intersection":1e-05,"epsilon_rel_max":0.001,"epsilon_step":1e-05,"errcon":0.0001,"max_nsteps":100,"max_stepping_decrease":0.1,"max_stepping_increase":5.0,"max_substeps":10,"minimum_step":1.0000000000000002e-06,"pgrow":-0.2,"pshrink":-0.25,"safety":0.9},"field":[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],"x":{"max":10.0,"min":-10.0,"num":2},"y":{"max":10.0,"min":-10.0,"num":2},"z":{"max":10.0,"min":-10.0,"num":2}})json";
            EXPECT_JSON_ROUND_TRIP(input, expected);
        }
    }
}

TEST(JsonIO, physics_import)
{
    {
        PhysicsImport input = [] {
            PhysicsFromFile pff;
            pff.input = "physics.root";
            return pff;
        }();

        static char const expected[]
            = R"json({"_type":"file","input":"physics.root"})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
    {
        PhysicsImport input = [] {
            PhysicsFromGeant pfg;
            pfg.ignore_processes = {"CoulombScat"};
            return pfg;
        }();

        static char const expected[]
            = R"json({"_type":"geant","data_selection":{"interpolation":{"bc":"geant","order":1,"type":"linear"}},"ignore_processes":["CoulombScat"]})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
}

TEST(JsonIO, scoring)
{
    Scoring input;
    input.simple_calo = [] {
        SimpleCalo sc;
        sc.volumes = {Label{"calo1"}, Label{"calo2"}};
        return sc;
    }();

    static char const expected[]
        = R"json({"simple_calo":{"volumes":["calo1","calo2"]}})json";
    EXPECT_JSON_ROUND_TRIP(input, expected);
}

TEST(JsonIO, setup_geant)
{
    // lar-sphere-cpu.inp.json from celer-sim/simple
    std::istringstream input{
        R"(
{
 "coulomb_scattering": false,
 "compton_scattering": true,
 "photoelectric": true,
 "rayleigh_scattering": true,
 "gamma_conversion": true,
 "gamma_general": false,
 "ionization": true,
 "annihilation": true,
 "brems": "all",
 "msc": "none",
 "eloss_fluctuation": true,
 "lpm": true,
 "optical": {
  "absorption": true,
  "rayleigh_scattering": true,
  "wavelength_shifting": {
   "enable": true,
   "time_profile": "exponential"
  },
  "wavelength_shifting2": {
   "enable": true,
   "time_profile": "exponential"
  }
 }
})"};

    GeantSetup gs;
    input >> gs;
    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        EXPECT_JSON_ROUND_TRIP(
            gs,
            R"json({"_format":"geant-physics","_units":"cgs","_version":"0.7.0","angle_limit_factor":1.0,"annihilation":true,"apply_cuts":false,"brems":"all","compton_scattering":true,"coulomb_scattering":false,"default_cutoff":0.1,"eloss_fluctuation":true,"em_bins_per_decade":7,"form_factor":"exponential","gamma_conversion":true,"gamma_general":false,"integral_approach":true,"ionization":true,"linear_loss_limit":0.01,"lowest_electron_energy":[0.001,"MeV"],"lowest_muhad_energy":[0.001,"MeV"],"lpm":true,"max_energy":[100000000.0,"MeV"],"min_energy":[0.0001,"MeV"],"msc":"none","msc_displaced":true,"msc_lambda_limit":0.1,"msc_muhad_displaced":false,"msc_muhad_range_factor":0.2,"msc_muhad_step_algorithm":"minimal","msc_range_factor":0.04,"msc_safety_factor":0.6,"msc_step_algorithm":"safety","msc_theta_limit":3.141592653589793,"mucf_physics":false,"muon":null,"optical":{"_format":"geant4-optical-physics","_version":"0.7.0","absorption":true,"boundary":{"invoke_sd":false},"cherenkov":{"max_beta_change":10.0,"max_photons":100,"stack_photons":true,"track_secondaries_first":true},"mie_scattering":true,"rayleigh_scattering":true,"scintillation":{"by_particle_type":false,"finite_rise_time":false,"stack_photons":true,"track_info":false,"track_secondaries_first":true},"verbose":false,"wavelength_shifting":{"time_profile":"exponential"},"wavelength_shifting2":{"time_profile":"exponential"}},"photoelectric":true,"rayleigh_scattering":true,"relaxation":"none","seltzer_berger_limit":[1000.0,"MeV"],"verbose":false})json");
    }
}

TEST(JsonIO, setup_geant_muon)
{
    GeantSetup::MuonSetup input;
    nlohmann::json::parse(R"json({"bremsstrahlung":true})json").get_to(input);

    EXPECT_JSON_ROUND_TRIP(
        input,
        R"json({"bremsstrahlung":true,"coulomb":false,"ionization":true,"msc":"urban","pair_production":true})json");

    // Null muon physics is represented as std::nullopt in GeantPhysicsOptions
}

TEST(JsonIO, setup_geant_optical)
{
    // Create with most processes disabled
    GeantSetup::OpticalSetup input;
    input.cherenkov = std::nullopt;
    input.scintillation = std::nullopt;
    input.wavelength_shifting = std::nullopt;
    input.wavelength_shifting2 = std::nullopt;
    input.rayleigh_scattering = false;
    input.mie_scattering = false;
    // absorption and boundary remain at defaults (enabled)

    EXPECT_JSON_ROUND_TRIP(
        input,
        R"json({"_format":"geant4-optical-physics","_version":"0.7.0","absorption":true,"boundary":{"invoke_sd":false},"cherenkov":null,"mie_scattering":false,"rayleigh_scattering":false,"scintillation":null,"verbose":false,"wavelength_shifting":null,"wavelength_shifting2":null})json");

    input = {};
    EXPECT_JSON_ROUND_TRIP(
        input,
        R"json({"_format":"geant4-optical-physics","_version":"0.7.0","absorption":true,"boundary":{"invoke_sd":false},"cherenkov":{"max_beta_change":10.0,"max_photons":100,"stack_photons":true,"track_secondaries_first":true},"mie_scattering":true,"rayleigh_scattering":true,"scintillation":{"by_particle_type":false,"finite_rise_time":false,"stack_photons":true,"track_info":false,"track_secondaries_first":true},"verbose":false,"wavelength_shifting":{"time_profile":"delta"},"wavelength_shifting2":{"time_profile":"delta"}})json");

    // Null optical physics is represented as std::nullopt in
    // GeantPhysicsOptions
}

TEST(JsonIO, optical_standalone_input)
{
    OpticalStandaloneInput input;
    input.problem.model.geometry = "geometry.gdml";
    input.problem.limits.steps = 1000;
    input.problem.limits.step_iters = 10000;

    static char const expected[]
        = R"json({"_format":"optical-standalone-input","_version":"0.7.0","geant_setup":{"_format":"geant4-optical-physics","_version":"0.7.0","absorption":true,"boundary":{"invoke_sd":false},"cherenkov":{"max_beta_change":10.0,"max_photons":100,"stack_photons":true,"track_secondaries_first":true},"mie_scattering":true,"rayleigh_scattering":true,"scintillation":{"by_particle_type":false,"finite_rise_time":false,"stack_photons":true,"track_info":false,"track_secondaries_first":true},"verbose":false,"wavelength_shifting":{"time_profile":"delta"},"wavelength_shifting2":{"time_profile":"delta"}},"problem":{"capacity":{"generators":null,"primaries":null,"tracks":null},"generator":{"_type":"em"},"limits":{"step_iters":10000,"steps":1000},"model":{"geometry":"geometry.gdml"},"output_file":"-","perfetto_file":null,"seed":0,"step":null,"timers":{"action":false,"step":false}},"system":{"device":null,"environment":{}}})json";
    EXPECT_JSON_ROUND_TRIP(input, expected);
}

TEST(JsonIO, standalone_input)
{
    StandaloneInput input;
    input.problem.model.geometry = "geometry.gdml";
    input.problem.tracking.limits.steps = 100;
    input.problem.tracking.limits.step_iters = 1000;
    input.problem.tracking.optical_limits.steps = 0;
    input.problem.tracking.optical_limits.step_iters = 0;
    input.events.generator = inp::ReadFileEvents{"events.json"};

    if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
    {
        static char const expected[]
            = R"json({"_format":"standalone-input","_version":"0.7.0","events":{"generator":{"_type":"read","event_file":"events.json"},"merge":false},"geant_setup":{"_format":"geant-physics","_units":"cgs","_version":"0.7.0","angle_limit_factor":1.0,"annihilation":true,"apply_cuts":false,"brems":"all","compton_scattering":true,"coulomb_scattering":false,"default_cutoff":0.1,"eloss_fluctuation":true,"em_bins_per_decade":7,"form_factor":"exponential","gamma_conversion":true,"gamma_general":false,"integral_approach":true,"ionization":true,"linear_loss_limit":0.01,"lowest_electron_energy":[0.001,"MeV"],"lowest_muhad_energy":[0.001,"MeV"],"lpm":true,"max_energy":[100000000.0,"MeV"],"min_energy":[0.0001,"MeV"],"msc":"urban","msc_displaced":true,"msc_lambda_limit":0.1,"msc_muhad_displaced":false,"msc_muhad_range_factor":0.2,"msc_muhad_step_algorithm":"minimal","msc_range_factor":0.04,"msc_safety_factor":0.6,"msc_step_algorithm":"safety","msc_theta_limit":3.141592653589793,"mucf_physics":false,"muon":null,"optical":null,"photoelectric":true,"rayleigh_scattering":true,"relaxation":"none","seltzer_berger_limit":[1000.0,"MeV"],"verbose":false},"physics_import":{"_type":"geant","data_selection":{"interpolation":{"bc":"geant","order":1,"type":"linear"}},"ignore_processes":[]},"problem":{"control":{"capacity":{"events":null,"initializers":null,"primaries":null,"secondaries":null,"tracks":null},"device_debug":null,"optical_capacity":null,"seed":0,"track_order":null,"warm_up":false},"diagnostics":{"action":false,"counters":{"event":true,"step":true},"export_files":{"geometry":"","offload":"","physics":""},"log_frequency":1,"mctruth":null,"output_file":"-","perfetto_file":"","slot":null,"status_checker":false,"step":null,"timers":{"action":false,"step":false}},"field":{"_type":"none"},"model":{"geometry":"geometry.gdml"},"scoring":{"simple_calo":null},"tracking":{"force_step_limit":0.0,"limits":{"field_substeps":10,"step_iters":1000,"steps":100},"optical_limits":{"step_iters":0,"steps":0}}},"system":{"device":null,"environment":{}}})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
}

TEST(JsonIO, system)
{
    System input;
    input.environment = {{"TWO", "2"}, {"ONE", "1"}};
    {
        // Without optional device
        static char const expected[]
            = R"json({"device":null,"environment":{"ONE":"1","TWO":"2"}})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }

    input.device.emplace();
    input.device->stack_size = 1024;
    input.device->heap_size = 8192;
    {
        // With device
        static char const expected[]
            = R"json({"device":{"heap_size":8192,"stack_size":1024},"environment":{"ONE":"1","TWO":"2"}})json";
        EXPECT_JSON_ROUND_TRIP(input, expected);
    }
}

TEST(JsonIO, tracking)
{
    Tracking input;
    input.limits.steps = 1000;
    input.limits.step_iters = 10000;
    input.limits.field_substeps = 100;
    input.optical_limits.steps = 0;
    input.optical_limits.step_iters = 0;

    static char const expected[]
        = R"json({"force_step_limit":0.0,"limits":{"field_substeps":100,"step_iters":10000,"steps":1000},"optical_limits":{"step_iters":0,"steps":0}})json";
    EXPECT_JSON_ROUND_TRIP(input, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace inp
}  // namespace celeritas
