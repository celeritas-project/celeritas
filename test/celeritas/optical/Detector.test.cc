//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Detector.test.cc
//---------------------------------------------------------------------------//
#include <numeric>
#include <utility>
#include <vector>

#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/inp/StandaloneInput.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/CoreState.hh"
#include "celeritas/optical/DetectorData.hh"
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/Types.hh"
#include "celeritas/optical/gen/DirectGeneratorAction.hh"
#include "celeritas/optical/gen/PrimaryGeneratorAction.hh"
#include "celeritas/optical/surface/SurfacePhysicsParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;

constexpr bool reference_configuration
    = ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
       && !CELERITAS_VECGEOM_SURFACE
       && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW);

//---------------------------------------------------------------------------//
/*!
 * Test optical detector and scoring.
 *
 * Because detectors are not directly loaded from GDML files, an override is
 * used for loading the detectors into the core parameters. The default optical
 * surface is set to be strictly transmitting to ensure hits are always
 * recorded.
 */
class DetectorTest : public ::celeritas::test::GeantTestBase
{
  public:
    std::string_view gdml_basename() const override { return "optical-box"; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto result = GeantTestBase::build_geant_options();
        result.optical = {};
        CELER_ENSURE(result.optical);
        return result;
    }

    GeantImportDataSelection build_import_data_selection() const override
    {
        auto result = GeantTestBase::build_import_data_selection();
        result.processes |= GeantImportDataSelection::optical;
        return result;
    }

    std::vector<IMC> select_optical_models() const override
    {
        return {IMC::absorption};
    }

    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        PhysSurfaceId phys_surface{0};

        inp::SurfacePhysics input;
        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.fresnel.emplace(phys_surface,
                                           inp::FresnelReflection{});
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::transmit);

        return std::make_shared<SurfacePhysicsParams>(
            this->optical_action_reg().get(), input);
    }

    SPConstDetectors detector() override
    {
        if (!detector_)
        {
            inp::Detectors input{{
                {"y-detectors", {VolumeId{1}, VolumeId{2}}},
                {"x-detectors", {VolumeId{3}, VolumeId{4}}},
                {"z-detectors", {VolumeId{5}, VolumeId{6}}},
            }};

            detector_ = std::make_shared<DetectorParams>(std::move(input),
                                                         *this->volume());
        }

        return detector_;
    }

    inp::OpticalDetector build_optical_detector_input() override
    {
        return detector_input_;
    }

    void initialize_run()
    {
        Transporter::Input inp;
        inp.params = this->optical_params();
        transport_ = std::make_shared<Transporter>(std::move(inp));

        size_type num_tracks = 128;
        state_ = std::make_shared<CoreState<MemSpace::host>>(
            *this->optical_params(), StreamId{0}, num_tracks);
        state_->aux() = std::make_shared<AuxStateVec>(
            *this->core()->aux_reg(), MemSpace::host, StreamId{0}, num_tracks);
    }

  protected:
    std::shared_ptr<CoreState<MemSpace::host>> state_;
    std::shared_ptr<AuxStateVec> aux_;
    std::shared_ptr<Transporter> transport_;
    std::shared_ptr<DetectorParams> detector_;

    inp::OpticalDetector detector_input_;
};

//---------------------------------------------------------------------------//
/*!
 * User-defined grid with non-zero efficiency on a surface to test detector
 * hits.
 */
class SurfaceDetectorTest : public DetectorTest
{
  public:
    SPConstOpticalSurfacePhysics build_optical_surface_physics() override
    {
        PhysSurfaceId phys_surface{0};

        inp::SurfacePhysics input;
        input.materials.push_back({});
        input.roughness.polished.emplace(phys_surface, inp::NoRoughness{});
        input.reflectivity.grid.emplace(phys_surface, [] {
            inp::GridReflection refl;
            std::vector<double> xs{1e-6, 2e-5};
            refl.reflectivity = inp::Grid{xs, {0.0, 0.0}};
            refl.transmittance = inp::Grid{xs, {0.0, 0.0}};
            refl.efficiency = inp::Grid{xs, {0.6, 0.6}};
            return refl;
        }());
        input.interaction.trivial.emplace(phys_surface,
                                          TrivialInteractionMode::transmit);

        return std::make_shared<SurfacePhysicsParams>(
            this->optical_action_reg().get(), input);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
// Run test to check small number of photons and hits to ensure correct hit
// information is populated.

struct SimpleScores
{
    std::vector<size_type> detector_ids;
    std::vector<real_type> energies;
    std::vector<real_type> times;
    std::vector<real_type> x_positions;
    std::vector<real_type> y_positions;
    std::vector<real_type> z_positions;
    std::vector<size_type> volume_instance_ids;
};

struct SimpleScorer
{
    SimpleScores& scores;

    void operator()(Span<DetectorHit const> new_hits)
    {
        for (auto const& hit : new_hits)
        {
            scores.detector_ids.push_back(hit.detector.unchecked_get());
            scores.energies.push_back(value_as<units::MevEnergy>(hit.energy));
            scores.times.push_back(hit.time);
            scores.x_positions.push_back(hit.position[0]);
            scores.y_positions.push_back(hit.position[1]);
            scores.z_positions.push_back(hit.position[2]);
            scores.volume_instance_ids.push_back(
                hit.volume_instance.unchecked_get());
        }
    }
};

TEST_F(DetectorTest, simple)
{
    SimpleScores scores;
    detector_input_.callback = SimpleScorer{scores};

    // Manually generate arbitrary photons aimed at different detectors

    using E = units::MevEnergy;
    using TI = TrackInitializer;

    std::vector<TI> const inits{
        TI{E{1e-6},
           Real3{0, 0, 0},  // pos
           Real3{1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           0,  // time
           {},
           ImplVolumeId{0}},
        TI{E{2e-6},
           Real3{0, 0, 0},  // pos
           Real3{-1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           10,  // time
           {},
           ImplVolumeId{0}},
        TI{E{3e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, 0, 1},  // dir
           Real3{0, 1, 0},  // pol
           1,  // time
           {},
           ImplVolumeId{0}},
        TI{E{4e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, 0, -1},  // dir
           Real3{0, 1, 0},  // pol
           20,  // time
           {},
           ImplVolumeId{0}},
        TI{E{5e-6},
           Real3{0, 0, 0},  // pos
           Real3{1, 0, 0},  // dir
           Real3{0, 1, 0},  // pol
           13,  // time
           {},
           ImplVolumeId{0}},
        TI{E{6e-6},
           Real3{0, 0, 0},  // pos
           Real3{0, -1, 0},  // dir
           Real3{1, 0, 0},  // pol
           7,  // time
           {},
           ImplVolumeId{0}},
    };

    // Run test

    auto generate
        = DirectGeneratorAction::make_and_insert(*this->optical_params());
    this->initialize_run();
    generate->insert(*state_, make_span(inits));
    (*transport_)(*state_);

    // Check results

    real_type const box_size = from_cm(50);
    real_type const flight_time = box_size / constants::c_light;

    static size_type const expected_detector_ids[] = {1, 1, 2, 2, 1, 0};
    static real_type const expected_energies[]
        = {1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6};
    static real_type const expected_x_positions[] = {
        box_size,
        -box_size,
        0,
        0,
        box_size,
        0,
    };
    static real_type const expected_y_positions[] = {
        0,
        0,
        0,
        0,
        0,
        -box_size,
    };
    static real_type const expected_z_positions[] = {
        0,
        0,
        box_size,
        -box_size,
        0,
        0,
    };
    static real_type const expected_times[] = {
        0 + flight_time,
        10 + flight_time,
        1 + flight_time,
        20 + flight_time,
        13 + flight_time,
        7 + flight_time,
    };
    static size_type const expected_volume_instance_ids[] = {5, 4, 6, 7, 5, 3};

    if (reference_configuration)
    {
        EXPECT_VEC_EQ(expected_detector_ids, scores.detector_ids);
        EXPECT_VEC_SOFT_EQ(expected_energies, scores.energies);
        EXPECT_VEC_SOFT_EQ(expected_x_positions, scores.x_positions);
        EXPECT_VEC_SOFT_EQ(expected_y_positions, scores.y_positions);
        EXPECT_VEC_SOFT_EQ(expected_z_positions, scores.z_positions);
        EXPECT_VEC_SOFT_EQ(expected_times, scores.times);
        EXPECT_VEC_EQ(expected_volume_instance_ids, scores.volume_instance_ids);
    }
}

//---------------------------------------------------------------------------//
// Run test over large number of photons to check buffering is done correctly.

struct StressScorer
{
    std::vector<size_type>& scores;
    size_type& errored;

    void operator()(Span<DetectorHit const> hits)
    {
        for (auto const& hit : hits)
        {
            if (hit.detector < scores.size())
            {
                scores[hit.detector.get()]++;
            }
            else
            {
                errored++;
            }
        }
    }
};

TEST_F(DetectorTest, stress)
{
    // 3 detectors: x, y, z
    std::vector<size_type> hits(3, 0);
    size_type errored = 0;
    detector_input_.callback = StressScorer{hits, errored};

    // Isotropically generate photons

    inp::OpticalPrimaryGenerator gen;
    gen.primaries = 8192;
    gen.energy = inp::MonoenergeticDistribution{1e-5};
    gen.angle = inp::IsotropicDistribution{};
    gen.shape = inp::PointDistribution{{0, 0, 0}};

    // Run test

    auto generate = PrimaryGeneratorAction::make_and_insert(
        *this->optical_params(), std::move(gen));
    this->initialize_run();
    generate->insert(*state_);
    (*transport_)(*state_);

    // Check results

    if (reference_configuration)
    {
        static size_type const expected_hits[] = {2673, 2816, 2703};

        EXPECT_VEC_EQ(expected_hits, hits);
        EXPECT_EQ(errored, 0);
    }
}

//---------------------------------------------------------------------------//
// Test surface efficiency propagates hits to detector
TEST_F(SurfaceDetectorTest, efficiency)
{
    // 3 detectors: x, y, z
    std::vector<size_type> hits(3, 0);
    size_type errored = 0;
    detector_input_.callback = StressScorer{hits, errored};

    // Isotropically generate photons

    inp::OpticalPrimaryGenerator gen;
    gen.primaries = 8192;
    gen.energy = inp::MonoenergeticDistribution{1e-5};
    gen.angle = inp::IsotropicDistribution{};
    gen.shape = inp::PointDistribution{{0, 0, 0}};

    // Run test

    auto generate = PrimaryGeneratorAction::make_and_insert(
        *this->optical_params(), std::move(gen));
    this->initialize_run();
    generate->insert(*state_);
    (*transport_)(*state_);

    // Check results

    if (reference_configuration)
    {
        auto total_hits = std::accumulate(hits.begin(), hits.end(), 0);

        // Expect ~60% of total primaries are detected
        static size_type const expected_hits = 4894;

        EXPECT_EQ(expected_hits, total_hits);
        EXPECT_EQ(errored, 0);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
