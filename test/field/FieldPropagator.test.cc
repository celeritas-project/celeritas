//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file FieldTrackView.test.cc
//---------------------------------------------------------------------------//
#include "FieldTestBase.hh"

#include "field/base/FieldTrackView.hh"
#include "field/FieldPropagator.hh"
#include "geometry/GeoStateStore.hh"

#ifdef CELERITAS_USE_CUDA
#    include "FieldPropagator.test.hh"
#endif

#include "field/MagField.hh"
#include "field/FieldEquation.hh"
#include "field/RungeKutta.hh"
#include "field/FieldIntegrator.hh"

using namespace celeritas_test;

//---------------------------------------------------------------------------//
// HOST TESTS
//---------------------------------------------------------------------------//
class FieldPropagatorHostTest : public FieldTestBase
{
  public:
    using Initializer_t = ParticleTrackView::Initializer_t;
};

TEST_F(FieldPropagatorHostTest, field_propagator_test_host)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView      geo(geo_params_view, geo_state_view, ThreadId(0));
    ParticleTrackView particle(
        particle_params->host_pointers(), state_ref, ThreadId(0));

    // Test input
    geo      = {{test_params.radius, -100, 0}, {0, 1, 0}};
    particle = Initializer_t{ParticleId{0}, MevEnergy{test_params.energy}};

    // Check GeoTrackView
    EXPECT_EQ(VolumeId{5}, geo.volume_id()); // World
    geo.find_next_step();
    EXPECT_SOFT_EQ(55.0, geo.next_step());

    // Check ParticleTrackView
    const ParticleParams& defs = *this->particle_params;
    EXPECT_EQ(ParticleId(0), defs.find(PDGNumber(11)));

    // Construct FieldTrackView
    FieldTrackView field_view(geo, particle);

    // Construct the RK stepper adnd propagator in a field
    const Real3     bfield{0, 0, test_params.field_value}; // a uniform B-field
    MagField        magfield(bfield);
    FieldEquation   equation(magfield);
    RungeKutta      rk4(equation);
    FieldIntegrator integrator(field_params_view, rk4);

    FieldPropagator propagator(field_params_view, integrator);

    // Initial state and the expected state after revolutions
    OdeArray yo;
    yo[0] = test_params.radius;
    yo[1] = -100;
    yo[4] = test_params.momentum;

    // Test parameters and the sub-step size
    double hstep = (2.0 * constants::pi * test_params.radius)
                   / test_params.nsteps;

    for (CELER_MAYBE_UNUSED int i : celeritas::range(test_params.revolutions))
    {
        for (CELER_MAYBE_UNUSED int j : celeritas::range(test_params.nsteps))
        {
            field_view.h() = hstep;
            real_type h    = propagator(field_view);
            EXPECT_DOUBLE_EQ(h, hstep);
        }
    }
    // Check input after num_revolutions
    EXPECT_VEC_NEAR(yo.get(), field_view.y().get(), 1.0e-3);

    // Test with boundary crossings
    GeoTrackView geo_0(geo_params_view, geo_state_view, ThreadId(0));
    geo_0 = {{test_params.radius, 0, 0}, {0, 1, 0}};
    EXPECT_EQ(VolumeId{2}, geo_0.volume_id());
    geo_0.find_next_step();
    EXPECT_SOFT_EQ(5.0, geo_0.next_step());
    FieldTrackView field_view_0(geo_0, particle);

    const int num_cross = 16;
    real_type y_cross[num_cross]
        = {5, 15, 25, 35, 35, 25, 15, 5, -5, -15, -25, -35, -35, -25, -15, -5};

    for (CELER_MAYBE_UNUSED int ir : celeritas::range(test_params.revolutions))
    {
        for (auto i : celeritas::range(num_cross))
        {
            field_view_0.h() = 2.0 * hstep;
            propagator(field_view_0);
            EXPECT_DOUBLE_EQ(y_cross[i], field_view_0.y()[1]);
        }
    }
}

#if CELERITAS_USE_CUDA
//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class FieldPropagatorDeviceTest : public FieldPropagatorHostTest
{
};

TEST_F(FieldPropagatorDeviceTest, field_propagator_device)
{
    CELER_ASSERT(geo_params);

    // Set up test input
    FPTestInput input;

    input.init_geo = {{{test_params.radius, -100, 0}, {0, 1, 0}},
                      {{test_params.radius, -100, 0}, {0, 1, 0}},
                      {{test_params.radius, -100, 0}, {0, 1, 0}},
                      {{test_params.radius, -100, 0}, {0, 1, 0}}};

    input.geo_params = geo_params->device_pointers();
    GeoStateStore device_states(*geo_params, input.init_geo.size());
    input.geo_states = device_states.device_pointers();

    // particle input
    input.init_track = {{ParticleId{0}, MevEnergy{test_params.energy}},
                        {ParticleId{0}, MevEnergy{test_params.energy}},
                        {ParticleId{0}, MevEnergy{test_params.energy}},
                        {ParticleId{0}, MevEnergy{test_params.energy}}};

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, input.init_track.size());
    input.particle_params = particle_params->device_pointers();
    input.particle_states = pstates.ref();

    input.field_params = this->field_params_view;
    input.test_params  = this->test_params;

    // Run kernel
    auto output = fp_test(input);

    // Check stepper results
    for (unsigned int i = 0; i < output.pos.size(); ++i)
    {
        EXPECT_LT(fabs(output.pos[i] - test_params.radius),
                  test_params.epsilon);
        EXPECT_LT(fabs(output.mom[i] - test_params.momentum),
                  test_params.epsilon);
        EXPECT_DOUBLE_EQ(output.step[i], 2392.9753773346288);
    }
}

TEST_F(FieldPropagatorDeviceTest, field_propagator_device_boundary)
{
    CELER_ASSERT(geo_params);

    // Set up test input
    FPTestInput input;

    input.init_geo = {{{test_params.radius, 0, 0}, {0, 1, 0}},
                      {{test_params.radius, 0, 0}, {0, 1, 0}},
                      {{test_params.radius, 0, 0}, {0, 1, 0}},
                      {{test_params.radius, 0, 0}, {0, 1, 0}}};

    input.geo_params = geo_params->device_pointers();
    GeoStateStore device_states(*geo_params, input.init_geo.size());
    input.geo_states = device_states.device_pointers();

    // particle input
    input.init_track = {{ParticleId{0}, MevEnergy{test_params.energy}},
                        {ParticleId{0}, MevEnergy{test_params.energy}},
                        {ParticleId{0}, MevEnergy{test_params.energy}},
                        {ParticleId{0}, MevEnergy{test_params.energy}}};

    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, input.init_track.size());
    input.particle_params = particle_params->device_pointers();
    input.particle_states = pstates.ref();

    input.field_params = this->field_params_view;
    input.test_params  = this->test_params;

    // Run kernel
    auto output = fp_test(input);

    // Check stepper results
    for (unsigned int i = 0; i < output.pos.size(); ++i)
    {
        EXPECT_LT(fabs(output.pos[i] + 28.731417961805139),
                  test_params.epsilon);
        EXPECT_LT(fabs(output.mom[i] + 8.6134826826429052),
                  test_params.epsilon);
        EXPECT_LT(fabs(output.step[i] - 1582.7054662411308),
                  test_params.epsilon);
    }
}

//---------------------------------------------------------------------------//
#endif
