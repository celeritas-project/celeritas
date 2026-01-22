//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/Generator.test.cc
//---------------------------------------------------------------------------//
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/random/distribution/PoissonDistribution.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/LArSphereBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/optical/CoreParams.hh"  // IWYU pragma: keep
#include "celeritas/optical/CoreState.hh"  // IWYU pragma: keep
#include "celeritas/optical/Transporter.hh"
#include "celeritas/optical/gen/DirectGeneratorAction.hh"
#include "celeritas/optical/gen/GeneratorAction.hh"
#include "celeritas/optical/gen/GeneratorData.hh"
#include "celeritas/optical/gen/OffloadData.hh"
#include "celeritas/optical/gen/PrimaryGeneratorAction.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

// Reference results:
// - Double precision
// - Not vecgeom surface
constexpr bool reference_configuration
    = ((CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_DOUBLE)
       && (CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_VECGEOM
           || !CELERITAS_VECGEOM_SURFACE)
       && CELERITAS_CORE_RNG == CELERITAS_CORE_RNG_XORWOW);

//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSphereGeneratorTest : public LArSphereBase
{
  public:
    using VecDistribution = std::vector<optical::GeneratorDistributionData>;

  public:
    void SetUp() override {}

    SPConstAction build_along_step() override
    {
        return LArSphereBase::build_along_step();
    }

    //! Construct the optical and aux state data
    template<MemSpace M>
    void build_state(size_type size)
    {
        if constexpr (M == MemSpace::device)
        {
            device().create_streams(1);
        }

        state_ = std::make_shared<optical::CoreState<M>>(
            *this->optical_params(), StreamId{0}, size);
        state_->aux() = std::make_shared<AuxStateVec>(
            *this->core()->aux_reg(), M, StreamId{0}, size);
    }

    //! Construct the optical transporter
    void build_transporter()
    {
        // Create transporter with aux data for accumulating action times
        optical::Transporter::Input inp;
        inp.params = this->optical_params();
        inp.action_times = ActionTimes::make_and_insert(
            this->optical_params()->action_reg(),
            this->core()->aux_reg(),
            "optical-action-times");
        transport_ = std::make_shared<optical::Transporter>(std::move(inp));
    }

    //! Buffer host distribution data for Cherenkov and scintillation
    VecDistribution make_distributions(size_type count, size_type& num_photons)
    {
        using GT = GeneratorType;

        std::vector<GT> types{GT::cherenkov, GT::scintillation};
        std::mt19937 rng;

        PoissonDistribution<real_type> sample_num_photons(100);

        optical::GeneratorDistributionData data;
        data.step_length = from_cm(0.2);
        data.charge = units::ElementaryCharge{-1};
        data.material = OptMatId(0);
        data.continuous_edep_fraction = 1;
        data.points[StepPoint::pre]
            = {units::LightSpeed(0.7), from_cm(Real3{0, 0, 0})};
        data.points[StepPoint::post]
            = {units::LightSpeed(0.6), from_cm(Real3{0, 0, 0.2})};

        VecDistribution result(count, data);
        for (auto i : range(count))
        {
            result[i].type = types[i % types.size()];
            result[i].num_photons = sample_num_photons(rng);
            num_photons += result[i].num_photons;
            CELER_ASSERT(result[i]);
        }
        return result;
    }

    //! Get optical counters
    OpticalAccumStats counters(optical::GeneratorBase const& gen) const
    {
        OpticalAccumStats result = state_->accum();
        result.generators.push_back(gen.counters(*state_->aux()).accum);
        return result;
    }

  protected:
    std::shared_ptr<optical::CoreStateBase> state_;
    std::shared_ptr<optical::Transporter> transport_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LArSphereGeneratorTest, primary_generator)
{
    // Create primary generator action
    inp::OpticalPrimaryGenerator inp;
    inp.primaries = 65536;
    inp.energy = inp::MonoenergeticDistribution{1e-5};
    inp.angle = inp::IsotropicDistribution{};
    inp.shape = inp::PointDistribution{{0, 0, 0}};
    auto generate = optical::PrimaryGeneratorAction::make_and_insert(
        *this->optical_params(), std::move(inp));

    this->build_transporter();
    this->build_state<MemSpace::host>(4096);

    // Queue primaries
    generate->insert(*state_);

    // Launch the optical loop
    (*transport_)(*state_);

    // Get the accumulated counters
    auto result = this->counters(*generate);

    if (reference_configuration)
    {
        EXPECT_EQ(68939, result.steps);
        EXPECT_EQ(18, result.step_iters);
    }
    EXPECT_EQ(1, result.flushes);
    ASSERT_EQ(1, result.generators.size());

    auto const& gen = result.generators.front();
    EXPECT_EQ(0, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(65536, gen.num_generated);
}

TEST_F(LArSphereGeneratorTest, TEST_IF_CELER_DEVICE(device_primary_generator))
{
    // Create primary generator action
    inp::OpticalPrimaryGenerator inp;
    inp.primaries = 65536;
    inp.energy = inp::NormalDistribution{1e-5, 1e-6};
    inp.angle = inp::MonodirectionalDistribution{{1, 0, 0}};
    inp.shape = inp::UniformBoxDistribution{{-10, -10, -10}, {10, 10, 10}};
    auto generate = optical::PrimaryGeneratorAction::make_and_insert(
        *this->optical_params(), std::move(inp));

    this->build_transporter();
    this->build_state<MemSpace::device>(16384);

    // Queue primaries
    generate->insert(*state_);

    // Launch the optical loop
    (*transport_)(*state_);

    // Get the accumulated counters
    auto result = this->counters(*generate);

    if (reference_configuration)
    {
        EXPECT_EQ(69257, result.steps);
        EXPECT_EQ(6, result.step_iters);
    }
    EXPECT_EQ(1, result.flushes);
    ASSERT_EQ(1, result.generators.size());

    auto const& gen = result.generators.front();
    EXPECT_EQ(0, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(65536, gen.num_generated);
}

TEST_F(LArSphereGeneratorTest, direct_generator)
{
    // Create direct generator action
    std::vector<optical::TrackInitializer> inits(
        128,
        optical::TrackInitializer{units::MevEnergy{1e-5},
                                  Real3{0, 0, 0},
                                  Real3{1, 0, 0},
                                  Real3{0, 1, 0},
                                  0,
                                  ImplVolumeId{0}});
    auto generate = optical::DirectGeneratorAction::make_and_insert(
        *this->optical_params());

    this->build_transporter();
    this->build_state<MemSpace::host>(32);

    // Queue primaries
    generate->insert(*state_, make_span(inits));

    // Launch the optical loop
    (*transport_)(*state_);

    // Get the accumulated counters
    auto result = this->counters(*generate);
    if (reference_configuration
        && CELERITAS_CORE_GEO != CELERITAS_CORE_GEO_GEANT4)
    {
        EXPECT_EQ(133, result.steps);
        EXPECT_EQ(5, result.step_iters);
    }
    EXPECT_EQ(1, result.flushes);
    ASSERT_EQ(1, result.generators.size());

    auto const& gen = result.generators.front();
    EXPECT_EQ(128, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);
    EXPECT_EQ(128, gen.num_generated);
}

TEST_F(LArSphereGeneratorTest, generator)
{
    // Create optical action to generate Cherenkov and scintillation photons
    size_type capacity = 512;
    auto generate = optical::GeneratorAction::make_and_insert(
        *this->optical_params(), capacity);

    this->build_transporter();
    this->build_state<MemSpace::host>(4096);

    // Create host distributions and copy to generator
    size_type num_photons{0};
    auto host_data = this->make_distributions(capacity, num_photons);
    state_->counters().num_pending = num_photons;
    generate->insert(*state_, make_span(host_data));

    // Launch the optical loop
    (*transport_)(*state_);

    // Get the accumulated counters
    auto result = this->counters(*generate);

    EXPECT_EQ(1, result.flushes);
    ASSERT_EQ(1, result.generators.size());

    auto const& gen = result.generators.front();
    EXPECT_EQ(512, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);

    if (reference_configuration)
    {
        EXPECT_EQ(51226, gen.num_generated);
        EXPECT_EQ(53429, result.steps);
        EXPECT_EQ(14, result.step_iters);
    }

    // Check accumulated action times
    std::set<std::string> labels;
    auto action_times = transport_->get_action_times(*state_->aux());
    for (auto const& [label, time] : action_times)
    {
        labels.insert(label);
        EXPECT_GT(time, 0);
    }
    static std::string const expected_labels[] = {
        "absorption",
        "along-step",
        "locate-vacancies",
        "optical-boundary-init",
        "optical-boundary-post",
        "optical-discrete-select",
        "optical-generate",
        "optical-surface-stepping",
        "pre-step",
        "tracking-cut",
    };
    EXPECT_VEC_EQ(expected_labels, labels);
}

TEST_F(LArSphereGeneratorTest, TEST_IF_CELER_DEVICE(device_generator))
{
    // Create optical action to generate Cherenkov and scintillation photons
    size_type capacity = 4096;
    auto generate = optical::GeneratorAction::make_and_insert(
        *this->optical_params(), capacity);

    this->build_transporter();
    this->build_state<MemSpace::device>(16384);

    // Create host distributions and copy to generator
    size_type num_photons{0};
    auto host_data = this->make_distributions(capacity, num_photons);
    state_->counters().num_pending = num_photons;
    generate->insert(*state_, make_span(host_data));

    // Launch the optical loop
    (*transport_)(*state_);

    // Get the accumulated counters
    auto result = this->counters(*generate);

    EXPECT_EQ(1, result.flushes);
    ASSERT_EQ(1, result.generators.size());

    auto const& gen = result.generators.front();
    EXPECT_EQ(4096, gen.buffer_size);
    EXPECT_EQ(0, gen.num_pending);

    if (reference_configuration)
    {
        EXPECT_EQ(409643, gen.num_generated);
        EXPECT_EQ(427544, result.steps);
        EXPECT_EQ(28, result.step_iters);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
