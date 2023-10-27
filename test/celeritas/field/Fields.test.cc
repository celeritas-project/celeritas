//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Fields.test.cc
//---------------------------------------------------------------------------//
#include <fstream>

#include "corecel/cont/Range.hh"
#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/field/UniformZField.hh"

#include "CMSParameterizedField.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST(UniformZFieldTest, all)
{
    UniformZField calc_field(123);

    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 123}), calc_field({100, -1, 0.5}));
}

TEST(UniformFieldTest, all)
{
    Real3 field_vec{1, 2, 3};
    UniformField calc_field(field_vec);

    EXPECT_VEC_SOFT_EQ(field_vec, calc_field({100, -1, 0.5}));
}

TEST(CMSParameterizedFieldTest, all)
{
    // Create the magnetic field with a parameterized field
    CMSParameterizedField calc_field;

    int const nsamples = 8;
    real_type delta_z = 25.0;
    real_type delta_r = 12.0;

    std::vector<real_type> actual;

    for (int i : range(nsamples))
    {
        Real3 field = calc_field(Real3{i * delta_r, i * delta_r, i * delta_z});
        for (real_type f : field)
        {
            actual.push_back(f / units::tesla);
        }
    }

    static const real_type expected_field[] = {-0,
                                               -0,
                                               3.8112023023834,
                                               0.00060945895519578,
                                               0.00060945895519578,
                                               3.8103569576023,
                                               0.0024581951993005,
                                               0.0024581951993005,
                                               3.8074692533866,
                                               0.0054638612329989,
                                               0.0054638612329989,
                                               3.8026007301972,
                                               0.0095877228523849,
                                               0.0095877228523849,
                                               3.7958506580647,
                                               0.014834624748597,
                                               0.014834624748597,
                                               3.7873486828586,
                                               0.021253065345318,
                                               0.021253065345318,
                                               3.7772444535824,
                                               0.028935543902684,
                                               0.028935543902684,
                                               3.7656950871883};
    EXPECT_VEC_SOFT_EQ(expected_field, actual);
}

#define RZMapFieldTest TEST_IF_CELERITAS_JSON(RZMapFieldTest)
class RZMapFieldTest : public ::celeritas::test::Test
{
};

TEST_F(RZMapFieldTest, all)
{
    RZMapFieldParams field_map = [this] {
        // Read input file from JSON
        RZMapFieldInput inp;
        auto filename
            = this->test_data_path("celeritas", "cms-tiny.field.json");
        std::ifstream(filename) >> inp;
        return RZMapFieldParams(inp);
    }();

    RZMapField calc_field(field_map.host_ref());

    int const nsamples = 8;
    real_type delta_z = 25.0;
    real_type delta_r = 12.0;

    std::vector<real_type> actual;

    for (int i : range(nsamples))
    {
        Real3 field = calc_field(Real3{i * delta_r, i * delta_r, i * delta_z});
        for (real_type f : field)
        {
            // Reference result is in [T]: convert from native units
            actual.push_back(f / units::tesla);
        }
    }

    static const real_type expected_field[] = {-0,
                                               -0,
                                               3.811202287674,
                                               -4.7522817039862e-05,
                                               -4.7522817039862e-05,
                                               3.8062113523483,
                                               -9.5045634079725e-05,
                                               -9.5045634079725e-05,
                                               3.8012204170227,
                                               -0.00014256845111959,
                                               -0.00014256845111959,
                                               3.7962294816971,
                                               0.0094939613342285,
                                               0.0094939613342285,
                                               3.7912385463715,
                                               0.011867451667786,
                                               0.011867451667786,
                                               3.775991499424,
                                               0.014240986622126,
                                               0.014240986622126,
                                               3.771880030632,
                                               0.016614892251046,
                                               0.016614892251046,
                                               3.757196366787};
    EXPECT_VEC_NEAR(expected_field, actual, real_type{1e-7});
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
