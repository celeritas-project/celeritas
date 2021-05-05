//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#include "FieldTestBase.hh"

#ifdef CELERITAS_USE_CUDA
#    include "FieldPropagator.test.hh"
#endif

#include "field/MagField.hh"
#include "field/MagFieldEquation.hh"
#include "field/RungeKuttaStepper.hh"
#include "field/FieldDriver.hh"
#include "field/FieldPropagator.hh"

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//

class FieldPropagatorHostTest : public FieldTestBase
{
  public:
    using Initializer_t = ParticleTrackView::Initializer_t;
};

TEST_F(FieldPropagatorHostTest, field_propagator_host)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView geo_track = GeoTrackView(
        this->geo_params->host_pointers(), geo_state.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        particle_params->host_pointers(), state_ref, ThreadId(0));

    // Construct FieldPropagator
    MagField         field({0, 0, test.field_value});
    MagFieldEquation equation(field, units::ElementaryCharge{-1});
    RungeKuttaStepper<MagFieldEquation> rk4(equation);
    FieldDriver                         driver(field_params, rk4);

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
        geo_track.find_next_step();
        EXPECT_SOFT_EQ(5.5, geo_track.next_step());

        // Construct FieldPropagator
        FieldPropagator propagator(&geo_track, particle_track, driver);

        real_type                    total_length = 0;
        FieldPropagator::result_type result;

        for (CELER_MAYBE_UNUSED int ir : celeritas::range(test.revolutions))
        {
            for (CELER_MAYBE_UNUSED int j : celeritas::range(test.nsteps))
            {
                result = propagator(step);
                EXPECT_DOUBLE_EQ(result.distance, step);
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
        this->geo_params->host_pointers(), geo_state.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        particle_params->host_pointers(), state_ref, ThreadId(0));

    // Construct FieldDriver
    MagField         field({0, 0, test.field_value});
    MagFieldEquation equation(field, units::ElementaryCharge{-1});
    RungeKuttaStepper<MagFieldEquation> rk4(equation);
    FieldDriver                         driver(field_params, rk4);

    const int num_boundary = 16;

    // clang-format off
    real_type expected_y[num_boundary] 
        = { 0.5,  1.5,  2.5,  3.5,  3.5,  2.5,  1.5,  0.5,
           -0.5, -1.5, -2.5, -3.5, -3.5, -2.5, -1.5, -0.5};
    // clang-format on

    // Test parameters and the sub-step size
    double step = (2.0 * constants::pi * test.radius) / test.nsteps;

    for (auto i : celeritas::range(test.nstates))
    {
        // Initialize GeoTrackView and ParticleTrackView
        geo_track      = {{test.radius, 0, i * 1.0e-6}, {0, 1, 0}};
        particle_track = Initializer_t{ParticleId{0}, MevEnergy{test.energy}};

        geo_track.find_next_step();
        EXPECT_SOFT_EQ(0.5, geo_track.next_step());

        // Construct FieldPropagator
        FieldPropagator propagator(&geo_track, particle_track, driver);

        int                          icross       = 0;
        real_type                    total_length = 0;
        FieldPropagator::result_type result;

        for (CELER_MAYBE_UNUSED int ir : celeritas::range(test.revolutions))
        {
            for (CELER_MAYBE_UNUSED auto k : celeritas::range(test.nsteps))
            {
                result = propagator(step);
                total_length += result.distance;

                if (result.on_boundary)
                {
                    icross++;
                    int j = (icross - 1) % num_boundary;
                    EXPECT_DOUBLE_EQ(expected_y[j], geo_track.pos()[1]);
                }
            }
        }

        // Check stepper results with boundary crossings
        EXPECT_SOFT_NEAR(geo_track.pos()[0], -0.13151242, test.epsilon);
        EXPECT_SOFT_NEAR(geo_track.dir()[1], -0.03452005, test.epsilon);
        EXPECT_SOFT_NEAR(total_length, 221.48171708, test.epsilon);
    }
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class FieldPropagatorDeviceTest : public FieldPropagatorHostTest
{
  public:
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::device>;
};

TEST_F(FieldPropagatorDeviceTest, field_propagator_device)
{
    CELER_ASSERT(geo_params);

    // Set up test input
    FPTestInput input;
    for (unsigned int i : celeritas::range(test.nstates))
    {
        input.init_geo.push_back({{test.radius, -10, i * 1.0e-6}, {0, 1, 0}});
        input.init_track.push_back({ParticleId{0}, MevEnergy{test.energy}});
    }
    input.geo_params = geo_params->device_pointers();
    GeoStateStore device_states(*geo_params, input.init_geo.size());
    input.geo_states = device_states.ref();

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, input.init_track.size());

    input.particle_params = particle_params->device_pointers();
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
    CELER_ASSERT(geo_params);

    // Set up test input
    FPTestInput input;
    for (unsigned int i : celeritas::range(test.nstates))
    {
        input.init_geo.push_back({{test.radius, 0, i * 1.0e-6}, {0, 1, 0}});
        input.init_track.push_back({ParticleId{0}, MevEnergy{test.energy}});
    }

    input.geo_params = geo_params->device_pointers();
    GeoStateStore device_states(*geo_params, input.init_geo.size());
    input.geo_states = device_states.ref();

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, input.init_track.size());

    input.particle_params = particle_params->device_pointers();
    input.particle_states = pstates.ref();

    input.field_params = this->field_params;
    input.test         = this->test;

    // Run kernel
    auto output = bc_test(input);

    // Check stepper results
    for (unsigned int i = 0; i < output.pos.size(); ++i)
    {
        EXPECT_SOFT_NEAR(output.pos[i], -0.13151242, test.epsilon);
        EXPECT_SOFT_NEAR(output.dir[i], -0.03452005, test.epsilon);
        EXPECT_SOFT_NEAR(output.step[i], 221.48171708, test.epsilon);
    }
}

//---------------------------------------------------------------------------//
#endif
