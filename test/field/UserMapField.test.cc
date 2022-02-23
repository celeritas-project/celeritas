//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UserMapField.test.cc
//---------------------------------------------------------------------------//
#include "field/detail/FieldMapData.hh"
#include "field/detail/MagFieldMap.hh"

#include "celeritas_test.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"

#include "UserField.test.hh"
#include "detail/CMSFieldMapReader.hh"
#include "detail/CMSMapField.hh"

using celeritas::detail::CMSMapField;
using celeritas::detail::MagFieldMap;

using namespace celeritas;
using namespace celeritas_test;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class UserMapFieldTest : public Test
{
  protected:
    void SetUp() override
    {
        // Construct MagFieldMap and save a reference to the host data
        std::string test_file
            = celeritas::Test::test_data_path("field", "cmsFieldMap.tiny");

        detail::FieldMapParameters params;
        params.delta_grid = units::meter;
        params.num_grid_r = 9 + 1;           //! [0:9]
        params.num_grid_z = 2 * 16 + 1;      //! [-16:16]
        params.offset_z   = real_type{1600}; //! 16 meters

        MagFieldMap::ReadMap load_map
            = detail::CMSFieldMapReader(params, test_file);

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
    UserFieldTestParams                  test_param_;
    std::shared_ptr<MagFieldMap>         map_;
    celeritas::detail::FieldMapNativeRef ref_;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(UserMapFieldTest, host_map_field)
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

//---------------------------------------------------------------------------//
// DEVICE TESTS
//---------------------------------------------------------------------------//

class UserMapFieldDeviceTest : public UserMapFieldTest
{
  public:
    celeritas::detail::FieldMapDeviceRef device_ref_;
};

TEST_F(UserMapFieldDeviceTest, TEST_IF_CELER_DEVICE(device_map_field))
{
    // Run kernel for the magnetic field with a mapped field
    this->device_ref_ = this->map_->device_ref();

    auto output = fieldmap_test(this->test_param_, this->device_ref_);

    for (unsigned int i : celeritas::range(this->test_param_.nsamples))
    {
        Real3 value{output.value_x[i], output.value_y[i], output.value_z[i]};
        EXPECT_VEC_NEAR(this->expected_by_map[i], value, 1.0e-6);
    }
}
