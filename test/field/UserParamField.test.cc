//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UserParamField.test.cc
//---------------------------------------------------------------------------//
#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"

#include "UserField.test.hh"
#include "detail/CMSParameterizedField.hh"

using celeritas::detail::CMSParameterizedField;

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UserParamFieldTest : public Test
{
  protected:
    void SetUp() override
    {
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
    UserFieldTestParams test_param_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UserParamFieldTest, host_parameterized_field)
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

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class UserParamFieldDeviceTest : public UserParamFieldTest
{
};

TEST_F(UserParamFieldDeviceTest,
       TEST_IF_CELERITAS_CUDA(device_parameterized_field))
{
    // Run kernel for the magnetic field with a parameterized field
    auto output = parameterized_field_test(this->test_param_);

    for (unsigned int i : celeritas::range(this->test_param_.nsamples))
    {
        Real3 value{output.value_x[i], output.value_y[i], output.value_z[i]};
        EXPECT_VEC_NEAR(this->expected_by_param[i], value, 1.0e-6);
    }
}
