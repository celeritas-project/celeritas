//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Fields.test.cc
//---------------------------------------------------------------------------//
#include "corecel/cont/Range.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/field/UniformZField.hh"
#include "celeritas/field/detail/CMSParameterizedField.hh"

#include "celeritas_test.hh"
#include "detail/CMSFieldMapReader.hh"
#include "detail/CMSMapField.hh"
#include "detail/FieldMapData.hh"
#include "detail/MagFieldMap.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

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
    Real3        field_vec{1, 2, 3};
    UniformField calc_field(field_vec);

    EXPECT_VEC_SOFT_EQ(field_vec, calc_field({100, -1, 0.5}));
}

TEST(CMSParameterizedFieldTest, all)
{
    // Create the magnetic field with a parameterized field
    celeritas_test::detail::CMSParameterizedField calc_field;

    const int nsamples = 8;
    real_type delta_z  = 25.0;
    real_type delta_r  = 12.0;

    static const Real3 expected[nsamples]
        = {{-0.000000, -0.000000, 3811.202302},
           {0.609459, 0.609459, 3810.356958},
           {2.458195, 2.458195, 3807.469253},
           {5.463861, 5.463861, 3802.600730},
           {9.587723, 9.587723, 3795.850658},
           {14.834625, 14.834625, 3787.348683},
           {21.253065, 21.253065, 3777.244454},
           {28.935544, 28.935544, 3765.695087}};

    for (int i : celeritas::range(nsamples))
    {
        // Get the field value at a given position
        Real3 pos{i * delta_r, i * delta_r, i * delta_z};
        EXPECT_VEC_NEAR(expected[i], calc_field(pos), 1.0e-6);
    }
}

TEST(CMSMapField, all)
{
    using celeritas_test::detail::CMSFieldMapReader;
    using celeritas_test::detail::CMSMapField;
    using celeritas_test::detail::FieldMapParameters;
    using celeritas_test::detail::MagFieldMap;

    std::unique_ptr<MagFieldMap> field_map;
    {
        FieldMapParameters params;
        params.delta_grid = units::meter;
        params.num_grid_r = 9 + 1;      //! [0:9]
        params.num_grid_z = 2 * 16 + 1; //! [-16:16]
        params.offset_z   = 16 * units::meter;

        CMSFieldMapReader load_map(params,
                                   celeritas_test::Test::test_data_path(
                                       "celeritas", "cmsFieldMap.tiny"));
        field_map = std::make_unique<MagFieldMap>(load_map);
    }

    CMSMapField calc_field(field_map->host_ref());

    const int nsamples = 8;
    real_type delta_z  = 25.0;
    real_type delta_r  = 12.0;

    static const Real3 expected[nsamples]
        = {{-0.000000, -0.000000, 3811.202288},
           {-0.0475228, -0.0475228, 3806.21},
           {-0.0950456, -0.0950456, 3801.22},
           {-0.1425684, -0.1425684, 3796.23},
           {9.49396, 9.49396, 3791.24},
           {11.86745, 11.86745, 3775.99},
           {14.241, 14.241, 3771.88},
           {16.6149, 16.6149, 3757.2}};

    for (int i : celeritas::range(nsamples))
    {
        // Get the field value at a given position
        Real3 pos{i * delta_r, i * delta_r, i * delta_z};
        EXPECT_VEC_NEAR(expected[i], calc_field(pos), 1.0e-6);
    }
}
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
