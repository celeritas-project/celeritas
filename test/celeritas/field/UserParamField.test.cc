//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UserParamField.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/FieldPropagator.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/MagFieldTraits.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "FieldPropagator.test.hh"
#include "FieldPropagatorTestBase.hh"
#include "UserField.test.hh"
#include "celeritas_test.hh"
#include "detail/CMSParameterizedField.hh"

using celeritas_test::detail::CMSParameterizedField;

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UserParamFieldTest : public FieldPropagatorTestBase
{
  public:
    using Initializer_t = ParticleTrackView::Initializer_t;
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

  protected:
    void SetUp() override
    {
        FieldPropagatorTestBase::SetUp();
        geo_state_ = GeoStateStore(*this->geometry(), 1);

        // Scale the test radius with the approximated center value of the
        // parameterized field (3.8 units::tesla)
        test.radius /= 3.8;

        // Test parameters
        test_param_.nsamples = 8;
        test_param_.delta_z  = 200 / test_param_.nsamples;
        test_param_.delta_r  = 100 / test_param_.nsamples;
    }

    const Real3 expected_by_param[8] = {{-0.000000, -0.000000, 3811.202302},
                                        {0.609459, 0.609459, 3810.356958},
                                        {2.458195, 2.458195, 3807.469253},
                                        {5.463861, 5.463861, 3802.600730},
                                        {9.587723, 9.587723, 3795.850658},
                                        {14.834625, 14.834625, 3787.348683},
                                        {21.253065, 21.253065, 3777.244454},
                                        {28.935544, 28.935544, 3765.695087}};

  protected:
    // Test input and parameters
    GeoStateStore       geo_state_;
    UserFieldTestParams test_param_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UserParamFieldTest, host_upf_value)
{
    // Create the magnetic field with a parameterized field
    CMSParameterizedField field;

    for (int i : celeritas::range(this->test_param_.nsamples))
    {
        // Get the field value at a given position
        Real3 pos{i * this->test_param_.delta_r,
                  i * this->test_param_.delta_r,
                  i * this->test_param_.delta_z};
        Real3 value = field(pos);
        EXPECT_VEC_NEAR(this->expected_by_param[i], value, 1.0e-6);
    }
}

TEST_F(UserParamFieldTest, host_upf_propagator)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView geo_track = GeoTrackView(
        this->geometry()->host_ref(), geo_state_.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        this->particle()->host_ref(), state_ref, ThreadId(0));

    // Construct FieldDriver with a user CMSParameterizedField
    CMSParameterizedField field;
    using MFTraits
        = MagFieldTraits<CMSParameterizedField, DormandPrinceStepper>;
    MFTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    MFTraits::Stepper_t  stepper(equation);
    MFTraits::Driver_t   driver(field_params, stepper);

    // Test parameters and the sub-step size
    double step = (2.0 * constants::pi * test.radius) / test.nsteps;

    particle_track = Initializer_t{ParticleId{0}, MevEnergy{test.energy}};
    OdeState beg_state;
    beg_state.mom                   = {0, test.momentum_y, 0};
    real_type expected_total_length = 2 * constants::pi * test.radius
                                      * test.revolutions;

    for (unsigned int i : celeritas::range(test.nstates).step(128u))
    {
        // Initial state and the expected state after each revolution
        geo_track     = {{test.radius, -10, i * 1.0e-6}, {0, 1, 0}};
        beg_state.pos = {test.radius, -10, i * 1.0e-6};

        // Check GeoTrackView
        EXPECT_SOFT_EQ(5.5, geo_track.find_next_step().distance);

        // Construct FieldPropagator
        MFTraits::Propagator_t propagate(driver, particle_track, &geo_track);

        real_type                           total_length = 0;
        MFTraits::Propagator_t::result_type result;

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
        EXPECT_SOFT_NEAR(total_length, expected_total_length, test.epsilon);
    }
}

TEST_F(UserParamFieldTest, host_upf_geolimited)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView geo_track = GeoTrackView(
        this->geometry()->host_ref(), geo_state_.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        this->particle()->host_ref(), state_ref, ThreadId(0));

    // Construct FieldDriver with a user CMSParameterizedField
    CMSParameterizedField field;
    using MFTraits
        = MagFieldTraits<CMSParameterizedField, DormandPrinceStepper>;
    MFTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    MFTraits::Stepper_t  stepper(equation);
    MFTraits::Driver_t   driver(field_params, stepper);

    static const real_type expected_y[] = {0.5, 0.5, -0.5, -0.5};
    const int num_boundary = sizeof(expected_y) / sizeof(real_type);

    // Test parameters and the sub-step size
    double step = (2.0 * constants::pi * test.radius) / test.nsteps;

    for (auto i : celeritas::range(test.nstates).step(128u))
    {
        // Initialize GeoTrackView and ParticleTrackView
        geo_track      = {{test.radius, 0, i * 1.0e-6}, {0, 1, 0}};
        particle_track = Initializer_t{ParticleId{0}, MevEnergy{test.energy}};

        EXPECT_SOFT_EQ(0.5, geo_track.find_next_step().distance);

        // Construct FieldPropagator
        MFTraits::Propagator_t propagate(driver, particle_track, &geo_track);

        int                                 icross       = 0;
        real_type                           total_length = 0;
        MFTraits::Propagator_t::result_type result;

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
                    geo_track.cross_boundary();
                }
            }
        }

        // Check stepper results with boundary crossings
        EXPECT_SOFT_NEAR(61.557571992378342, total_length, test.epsilon);
    }
}

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//
#define UserParamFieldDeviceTest TEST_IF_CELER_DEVICE(UserParamFieldDeviceTest)
class UserParamFieldDeviceTest : public UserParamFieldTest
{
  public:
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::device>;
};

TEST_F(UserParamFieldDeviceTest, TEST_IF_CELER_DEVICE(device_upf_value))
{
    // Run kernel for the magnetic field with a parameterized field
    auto output = parameterized_field_test(this->test_param_);

    for (unsigned int i : celeritas::range(this->test_param_.nsamples))
    {
        Real3 value{output.value_x[i], output.value_y[i], output.value_z[i]};
        EXPECT_VEC_NEAR(this->expected_by_param[i], value, 1.0e-6);
    }
}

TEST_F(UserParamFieldDeviceTest, TEST_IF_CELER_DEVICE(device_upf_propagator))
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
        *this->particle(), input.init_track.size());

    input.particle_params = this->particle()->device_ref();
    input.particle_states = pstates.ref();

    input.field_params = this->field_params;
    input.test         = this->test;

    // Run kernel
    auto step = par_fp_test(input);

    // Check stepper results
    real_type step_length = 2 * constants::pi * test.radius * test.revolutions;
    for (unsigned int i = 0; i < step.size(); ++i)
    {
        EXPECT_SOFT_NEAR(step[i], step_length, test.epsilon);
    }
}

TEST_F(UserParamFieldDeviceTest, TEST_IF_CELER_DEVICE(device_upf_geolimited))
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
        *this->particle(), input.init_track.size());

    input.particle_params = this->particle()->device_ref();
    input.particle_states = pstates.ref();

    input.field_params = this->field_params;
    input.test         = this->test;

    // Run kernel
    auto step = par_bc_test(input);

    // Check stepper results
    for (unsigned int i = 0; i < step.size(); ++i)
    {
        EXPECT_SOFT_NEAR(step[i], 61.557571977595295, test.epsilon);
    }
}
