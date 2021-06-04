//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UserField.test.cc
//---------------------------------------------------------------------------//
#include "field/detail/FieldMapInterface.hh"
#include "field/detail/MagFieldMap.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"

#include "UserField.test.hh"
#include "detail/CMSFieldMapReader.hh"
#include "detail/CMSMapField.hh"
#include "detail/CMSParameterizedField.hh"

using celeritas::detail::CMSMapField;
using celeritas::detail::CMSParameterizedField;
using celeritas::detail::MagFieldMap;

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UserFieldTest : public Test
{
  protected:
    void SetUp() override
    {
        // Construct MagFieldMap and set the host data group
        MagFieldMap::ReadMap load_map = detail::CMSFieldMapReader();

        map_   = std::make_shared<MagFieldMap>(load_map);
        group_ = map_->host_group();

        // Test parameters
        test_param_.nsamples = 8;
        test_param_.delta_z  = 200 / test_param_.nsamples;
        test_param_.delta_r  = 100 / test_param_.nsamples;
    }

    const Real3 expected_by_map[8] = {{-0.000000, -0.000000, 3811.202288},
                                      {0.609459, 0.609459, 3810.327053},
                                      {2.458195, 2.458195, 3807.409525},
                                      {5.463859, 5.463859, 3802.511454},
                                      {9.587717, 9.587717, 3795.731544},
                                      {14.834614, 14.834614, 3787.198544},
                                      {21.253048, 21.253048, 3777.061939},
                                      {27.302565, 27.302565, 3765.115023}};

    const Real3 expected_by_param[8] = {{-0.000000, -0.000000, 3811.202302},
                                        {0.609459, 0.609459, 3810.356958},
                                        {2.458195, 2.458195, 3807.469253},
                                        {5.463861, 5.463861, 3802.600730},
                                        {9.587723, 9.587723, 3795.850658},
                                        {14.834625, 14.834625, 3787.348683},
                                        {21.253065, 21.253065, 3777.244454},
                                        {28.935544, 28.935544, 3765.695087}};

  protected:
    UserFieldTestParams                  test_param_;
    std::shared_ptr<MagFieldMap>         map_;
    celeritas::detail::FieldMapNativeRef group_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UserFieldTest, host_map_field)
{
    // Create the magnetic field with a mapped field
    CMSMapField field(this->group_);

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

TEST_F(UserFieldTest, host_parameterized_field)
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

class UserFieldDeviceTest : public UserFieldTest
{
  public:
    celeritas::detail::FieldMapDeviceRef device_group_;
};

TEST_F(UserFieldDeviceTest, TEST_IF_CELERITAS_CUDA(device_map_field))
{
    // Run kernel for the magnetic field with a mapped field
    this->device_group_ = this->map_->device_group();

    auto output = fieldmap_test(this->test_param_, this->device_group_);

    for (unsigned int i : celeritas::range(this->test_param_.nsamples))
    {
        Real3 value{output.value_x[i], output.value_y[i], output.value_z[i]};
        EXPECT_VEC_NEAR(this->expected_by_map[i], value, 1.0e-6);
    }
}

TEST_F(UserFieldDeviceTest, TEST_IF_CELERITAS_CUDA(device_parameterized_field))
{
    // Run kernel for the magnetic field with a parameterized field
    auto output = parameterized_field_test(this->test_param_);

    for (unsigned int i : celeritas::range(this->test_param_.nsamples))
    {
        Real3 value{output.value_x[i], output.value_y[i], output.value_z[i]};
        EXPECT_VEC_NEAR(this->expected_by_param[i], value, 1.0e-6);
    }
}
