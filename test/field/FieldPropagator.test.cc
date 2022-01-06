//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#include "field/FieldPropagator.hh"

#include "base/CollectionStateStore.hh"
#include "geometry/GeoData.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoTrackView.hh"
#include "physics/base/ParticleParams.hh"
#include "physics/base/ParticleData.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/Units.hh"
#include "field/FieldDriver.hh"
#include "field/FieldParamsData.hh"
#include "field/MagFieldEquation.hh"
#include "field/MagFieldTraits.hh"
#include "field/RungeKuttaStepper.hh"
#include "field/UniformMagField.hh"

#include "celeritas_test.hh"
#include "geometry/GeoTestBase.hh"
#include "FieldTestParams.hh"
#include "FieldPropagator.test.hh"

using namespace celeritas;
using namespace celeritas_test;
using celeritas::units::MevEnergy;

//---------------------------------------------------------------------------//
/*!
 * Test harness.
 *
 * The test geometry is 1-cm thick slabs normal to y, with 1-cm gaps in
 * between. We fire off electrons (TODO: also test positrons!) that end up
 * running circles around the z axis (along which the magnetic field points).
 */
class FieldPropagatorTestBase : public GeoTestBase<celeritas::GeoParams>
{
  public:
    const char* dirname() const override { return "field"; }
    const char* filebase() const override { return "field-test"; }

    void SetUp() override
    {
        using namespace celeritas::units;
        namespace pdg = celeritas::pdg;

        // Create particle defs
        constexpr auto        stable = ParticleDef::stable_decay_constant();
        ParticleParams::Input defs;
        defs.push_back({"electron",
                        pdg::electron(),
                        MevMass{0.5109989461},
                        ElementaryCharge{-1},
                        stable});
        defs.push_back({"positron",
                        pdg::positron(),
                        MevMass{0.5109989461},
                        ElementaryCharge{1},
                        stable});

        particle_params = std::make_shared<ParticleParams>(std::move(defs));

        // Construct views
        resize(&state_value, particle_params->host_ref(), 1);
        state_ref = state_value;

        // Set values of FieldParamsData;
        field_params.delta_intersection = 1.0e-4 * units::millimeter;

        // Input parameters of an electron in a uniform magnetic field
        test.nstates     = 128;
        test.nsteps      = 100;
        test.revolutions = 10;
        test.field_value = 1.0 * units::tesla;
        test.radius      = 3.8085386036;
        test.delta_z     = 6.7003310629;
        test.energy      = 10.9181415106;
        test.momentum_y  = 11.4177114158018;
        test.momentum_z  = 0.0;
        test.epsilon     = 1.0e-5;
    }

  protected:
    std::shared_ptr<ParticleParams>                         particle_params;
    ParticleStateData<Ownership::value, MemSpace::host>     state_value;
    ParticleStateData<Ownership::reference, MemSpace::host> state_ref;

    FieldParamsData field_params;

    // Test parameters
    FieldTestParams test;
};

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class FieldPropagatorHostTest : public FieldPropagatorTestBase
{
  public:
    using Initializer_t = ParticleTrackView::Initializer_t;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

    void SetUp()
    {
        FieldPropagatorTestBase::SetUp();
        geo_state_ = GeoStateStore(*this->geometry(), 1);
    }

  protected:
    GeoStateStore geo_state_;
};

TEST_F(FieldPropagatorHostTest, field_propagator_host)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView geo_track = GeoTrackView(
        this->geometry()->host_ref(), geo_state_.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        particle_params->host_ref(), state_ref, ThreadId(0));

    // Construct FieldPropagator
    UniformMagField field({0, 0, test.field_value});
    using RKTraits = MagFieldTraits<UniformMagField, RungeKuttaStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);
    RKTraits::Driver_t   driver(field_params, &rk4);

    // Test parameters and the sub-step size
    double step = (2.0 * constants::pi * test.radius) / test.nsteps;

    particle_track = Initializer_t{ParticleId{0}, MevEnergy{test.energy}};
    OdeState beg_state;
    beg_state.mom                   = {0, test.momentum_y, 0};
    real_type expected_total_length = 2 * constants::pi * test.radius
                                      * test.revolutions;

    for (unsigned int i : celeritas::range(test.nstates))
    {
        // Initial state and the expected state after each revolution
        geo_track     = {{test.radius, -10, i * 1.0e-6}, {0, 1, 0}};
        beg_state.pos = {test.radius, -10, i * 1.0e-6};

        // Check GeoTrackView
        EXPECT_SOFT_EQ(5.5, geo_track.find_next_step());

        // Construct FieldPropagator
        RKTraits::Propagator_t propagate(particle_track, &geo_track, &driver);

        real_type                           total_length = 0;
        RKTraits::Propagator_t::result_type result;

        for (CELER_MAYBE_UNUSED int ir : celeritas::range(test.revolutions))
        {
            for (CELER_MAYBE_UNUSED int j : celeritas::range(test.nsteps))
            {
                result = propagate(step);
                EXPECT_FALSE(result.boundary);
                EXPECT_DOUBLE_EQ(step, result.distance);
                total_length += result.distance;
            }
        }

        // Check input after num_revolutions
        EXPECT_VEC_NEAR(beg_state.pos, geo_track.pos(), test.epsilon);
        Real3 final_dir = beg_state.mom;
        normalize_direction(&final_dir);
        EXPECT_VEC_NEAR(final_dir, geo_track.dir(), test.epsilon);
        EXPECT_SOFT_NEAR(total_length, expected_total_length, test.epsilon);
    }
}

TEST_F(FieldPropagatorHostTest, boundary_crossing_host)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView geo_track = GeoTrackView(
        this->geometry()->host_ref(), geo_state_.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        particle_params->host_ref(), state_ref, ThreadId(0));

    // Construct FieldDriver
    UniformMagField field({0, 0, test.field_value});
    using RKTraits = MagFieldTraits<UniformMagField, RungeKuttaStepper>;
    RKTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    RKTraits::Stepper_t  rk4(equation);
    RKTraits::Driver_t   driver(field_params, &rk4);

    // clang-format off
    static const real_type expected_y[]
        = { 0.5,  1.5,  2.5,  3.5,  3.5,  2.5,  1.5,  0.5,
           -0.5, -1.5, -2.5, -3.5, -3.5, -2.5, -1.5, -0.5};
    // clang-format on
    const int num_boundary = sizeof(expected_y) / sizeof(real_type);

    // Test parameters and the sub-step size
    double step = (2.0 * constants::pi * test.radius) / test.nsteps;

    for (auto i : celeritas::range(test.nstates))
    {
        // Initialize GeoTrackView and ParticleTrackView
        geo_track      = {{test.radius, 0, i * 1.0e-6}, {0, 1, 0}};
        particle_track = Initializer_t{ParticleId{0}, MevEnergy{test.energy}};

        EXPECT_SOFT_EQ(0.5, geo_track.find_next_step());

        // Construct FieldPropagator
        RKTraits::Propagator_t propagate(particle_track, &geo_track, &driver);

        int                                 icross       = 0;
        real_type                           total_length = 0;
        RKTraits::Propagator_t::result_type result;

        for (CELER_MAYBE_UNUSED int ir : celeritas::range(test.revolutions))
        {
            for (CELER_MAYBE_UNUSED auto k : celeritas::range(test.nsteps))
            {
                result = propagate(step);
                total_length += result.distance;

                if (result.boundary)
                {
                    icross++;
                    int j = (icross - 1) % num_boundary;
                    EXPECT_DOUBLE_EQ(expected_y[j], geo_track.pos()[1]);
                }
            }
        }

        // Check stepper results with boundary crossings
        EXPECT_SOFT_NEAR(-0.13150565, geo_track.pos()[0], test.epsilon);
        EXPECT_SOFT_NEAR(-0.03453068, geo_track.dir()[1], test.epsilon);
        EXPECT_SOFT_NEAR(221.48171708, total_length, test.epsilon);
    }
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

#define FieldPropagatorDeviceTest \
    TEST_IF_CELERITAS_CUDA(FieldPropagatorDeviceTest)
class FieldPropagatorDeviceTest : public FieldPropagatorTestBase
{
  public:
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::device>;
};

TEST_F(FieldPropagatorDeviceTest, field_propagator_device)
{
    // Set up test input
    FPTestInput input;
    for (unsigned int i : celeritas::range(test.nstates))
    {
        input.init_geo.push_back({{test.radius, -10, i * 1.0e-6}, {0, 1, 0}});
        input.init_track.push_back({ParticleId{0}, MevEnergy{test.energy}});
    }
    input.geo_params = this->geometry()->device_ref();
    GeoStateStore device_states(*this->geometry(), input.init_geo.size());
    input.geo_states = device_states.ref();

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, input.init_track.size());

    input.particle_params = particle_params->device_ref();
    input.particle_states = pstates.ref();

    input.field_params = this->field_params;
    input.test         = this->test;

    // Run kernel
    auto output = fp_test(input);

    // Check stepper results
    real_type step_length = 2 * constants::pi * test.radius * test.revolutions;
    for (unsigned int i = 0; i < output.pos.size(); ++i)
    {
        EXPECT_SOFT_NEAR(output.pos[i], test.radius, test.epsilon);
        EXPECT_SOFT_NEAR(output.dir[i], 1.0, test.epsilon);
        EXPECT_SOFT_NEAR(output.step[i], step_length, test.epsilon);
    }
}

TEST_F(FieldPropagatorDeviceTest, boundary_crossing_device)
{
    // Set up test input
    FPTestInput input;
    for (unsigned int i : celeritas::range(test.nstates))
    {
        input.init_geo.push_back({{test.radius, 0, i * 1.0e-6}, {0, 1, 0}});
        input.init_track.push_back({ParticleId{0}, MevEnergy{test.energy}});
    }

    input.geo_params = this->geometry()->device_ref();
    GeoStateStore device_states(*this->geometry(), input.init_geo.size());
    input.geo_states = device_states.ref();

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, input.init_track.size());

    input.particle_params = particle_params->device_ref();
    input.particle_states = pstates.ref();

    input.field_params = this->field_params;
    input.test         = this->test;

    // Run kernel
    auto output = bc_test(input);

    // Check stepper results
    for (unsigned int i = 0; i < output.pos.size(); ++i)
    {
        EXPECT_SOFT_NEAR(output.pos[i], -0.13150565, test.epsilon);
        EXPECT_SOFT_NEAR(output.dir[i], -0.03453068, test.epsilon);
        EXPECT_SOFT_NEAR(output.step[i], 221.48171708, test.epsilon);
    }
}
