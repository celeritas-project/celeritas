//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UserMapField.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/field/DormandPrinceStepper.hh"
#include "celeritas/field/FieldDriver.hh"
#include "celeritas/field/FieldDriverOptions.hh"
#include "celeritas/field/MagFieldEquation.hh"
#include "celeritas/field/MagFieldTraits.hh"
#include "celeritas/geo/GeoParams.hh"

#include "FieldPropagatorTestBase.hh"
#include "UserField.test.hh"
#include "celeritas_test.hh"
#include "detail/CMSFieldMapReader.hh"
#include "detail/CMSMapField.hh"
#include "detail/FieldMapData.hh"
#include "detail/MagFieldMap.hh"

using celeritas_test::detail::CMSFieldMapReader;
using celeritas_test::detail::CMSMapField;
using celeritas_test::detail::FieldMapParameters;
using celeritas_test::detail::FieldMapRef;
using celeritas_test::detail::MagFieldMap;

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UserMapFieldTest : public FieldPropagatorTestBase
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
        // test field map (3.8 units::tesla)
        test.radius /= 3.8;

        // Construct MagFieldMap and save a reference to the host data
        std::string test_file = celeritas_test::Test::test_data_path(
            "celeritas", "cmsFieldMap.tiny");

        FieldMapParameters params;
        params.delta_grid = units::meter;
        params.num_grid_r = 9 + 1;           //! [0:9]
        params.num_grid_z = 2 * 16 + 1;      //! [-16:16]
        params.offset_z   = real_type{1600}; //! 16 meters

        MagFieldMap::ReadMap load_map = CMSFieldMapReader(params, test_file);

        map_ = std::make_shared<MagFieldMap>(load_map);
        ref_ = map_->host_ref();

        // Test parameters
        test_param_.nsamples = 8;
        test_param_.delta_z  = 200 / test_param_.nsamples;
        test_param_.delta_r  = 100 / test_param_.nsamples;
    }

    const Real3 expected_by_map[8] = {{-0.000000, -0.000000, 3811.202288},
                                      {-0.0475228, -0.0475228, 3806.21},
                                      {-0.0950456, -0.0950456, 3801.22},
                                      {-0.1425684, -0.1425684, 3796.23},
                                      {9.49396, 9.49396, 3791.24},
                                      {11.86745, 11.86745, 3775.99},
                                      {14.241, 14.241, 3771.88},
                                      {16.6149, 16.6149, 3757.2}};

  protected:
    // Test parameters and input
    GeoStateStore                  geo_state_;
    UserFieldTestParams            test_param_;
    std::shared_ptr<MagFieldMap>   map_;
    FieldMapRef                    ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UserMapFieldTest, host_umf_value)
{
    // Create the magnetic field with a mapped field
    CMSMapField field(this->ref_);

    for (int i : celeritas::range(this->test_param_.nsamples))
    {
        // Get the field value at a given position
        Real3 pos{i * this->test_param_.delta_r,
                  i * this->test_param_.delta_r,
                  i * this->test_param_.delta_z};
        Real3 value = field(pos);
        EXPECT_VEC_NEAR(this->expected_by_map[i], value, 1.0e-6);
    }
}

TEST_F(UserMapFieldTest, host_umf_propagator)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView geo_track = GeoTrackView(
        this->geometry()->host_ref(), geo_state_.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        this->particle()->host_ref(), state_ref, ThreadId(0));

    // Construct FieldDriver with a user CMSMapField
    CMSMapField field(this->ref_);

    using MFTraits = MagFieldTraits<CMSMapField, DormandPrinceStepper>;
    MFTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    MFTraits::Stepper_t  stepper(equation);
    MFTraits::Driver_t   driver(field_params, &stepper);

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
        MFTraits::Propagator_t propagate(particle_track, &geo_track, &driver);

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

TEST_F(UserMapFieldTest, host_umf_geolimited)
{
    // Construct GeoTrackView and ParticleTrackView
    GeoTrackView geo_track = GeoTrackView(
        this->geometry()->host_ref(), geo_state_.ref(), ThreadId(0));
    ParticleTrackView particle_track(
        this->particle()->host_ref(), state_ref, ThreadId(0));

    // Construct FieldDriver with a user CMSMapField
    CMSMapField field(this->ref_);
    using MFTraits = MagFieldTraits<CMSMapField, DormandPrinceStepper>;
    MFTraits::Equation_t equation(field, units::ElementaryCharge{-1});
    MFTraits::Stepper_t  stepper(equation);
    MFTraits::Driver_t   driver(field_params, &stepper);

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
        MFTraits::Propagator_t propagate(particle_track, &geo_track, &driver);

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
#define UserMapFieldDeviceTest TEST_IF_CELER_DEVICE(UserMapFieldDeviceTest)
class UserMapFieldDeviceTest : public UserMapFieldTest
{
  public:
    using GeoStateStore = CollectionStateStore<GeoStateData, MemSpace::device>;
};

TEST_F(UserMapFieldDeviceTest, TEST_IF_CELER_DEVICE(device_umf_value))
{
    // Run kernel for the magnetic field with a mapped field
    auto output = fieldmap_test(this->test_param_, this->map_->device_ref());

    for (unsigned int i : celeritas::range(this->test_param_.nsamples))
    {
        Real3 value{output.value_x[i], output.value_y[i], output.value_z[i]};
        EXPECT_VEC_NEAR(this->expected_by_map[i], value, 1.0e-6);
    }
}

TEST_F(UserMapFieldDeviceTest, TEST_IF_CELER_DEVICE(device_umf_propagator))
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
    auto step = map_fp_test(input, this->map_->device_ref());

    // Check stepper results
    real_type step_length = 2 * constants::pi * test.radius * test.revolutions;
    for (unsigned int i = 0; i < step.size(); ++i)
    {
        EXPECT_SOFT_NEAR(step[i], step_length, test.epsilon);
    }
}

TEST_F(UserMapFieldDeviceTest, TEST_IF_CELER_DEVICE(device_umf_geolimited))
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
    auto step = map_bc_test(input, this->map_->device_ref());

    // Check stepper results
    for (unsigned int i = 0; i < step.size(); ++i)
    {
        EXPECT_SOFT_NEAR(step[i], 61.557571977595295, test.epsilon);
    }
}
