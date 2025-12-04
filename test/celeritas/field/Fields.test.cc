//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/Fields.test.cc
//---------------------------------------------------------------------------//
#include "Fields.test.hh"

#include <algorithm>
#include <fstream>

#include "corecel/Config.hh"

#include "corecel/Constants.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/grid/GridTypes.hh"
#include "corecel/grid/Interpolator.hh"
#include "corecel/math/Quantity.hh"
#include "corecel/math/Turn.hh"
#include "geocel/Types.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"
#include "celeritas/field/CylMapField.hh"
#include "celeritas/field/CylMapFieldInput.hh"
#include "celeritas/field/CylMapFieldParams.hh"
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

    EXPECT_VEC_SOFT_EQ((Real3{0, 0, 123}),
                       calc_field(from_cm(Real3{100, -1, 0.5})));
}

TEST(UniformFieldTest, all)
{
    Real3 field_vec{1, 2, 3};
    UniformField calc_field(field_vec);

    EXPECT_VEC_SOFT_EQ(field_vec, calc_field(from_cm(Real3{100, -1, 0.5})));
}

TEST(CMSParameterizedFieldTest, all)
{
    // Create the magnetic field with a parameterized field
    CMSParameterizedField calc_field;

    int const nsamples = 8;
    real_type const delta_z = from_cm(25.0);
    real_type const delta_r = from_cm(12.0);

    std::vector<real_type> actual;

    for (int i : range(nsamples))
    {
        Real3 field = calc_field(Real3{i * delta_r, i * delta_r, i * delta_z});
        for (real_type f : field)
        {
            actual.push_back(native_value_to<units::FieldTesla>(f).value());
        }
    }

    static real_type const expected_field[] = {
        -0,
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
        3.7656950871883,
    };
    EXPECT_VEC_SOFT_EQ(expected_field, actual);
}

using RZMapFieldTest = ::celeritas::test::Test;

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
    real_type delta_z = from_cm(25.0);
    real_type delta_r = from_cm(12.0);

    std::vector<real_type> actual;

    for (int i : range(nsamples))
    {
        Real3 field = calc_field(Real3{i * delta_r, i * delta_r, i * delta_z});
        for (real_type f : field)
        {
            // Reference result is in [T]: convert from native units
            actual.push_back(native_value_to<units::FieldTesla>(f).value());
        }
    }

    static real_type const expected_field[] = {
        -0,
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
        3.757196366787,
    };
    EXPECT_VEC_NEAR(expected_field, actual, real_type{1e-7});
}

using CylMapFieldTest = ::celeritas::test::Test;

TEST_F(CylMapFieldTest, all)
{
    CylMapFieldParams field_map = [] {
        CylMapFieldInput inp;
        // Set up grid points in cylindrical coordinates
        inp.grid_r = {0, 50, 100, 150};
        Array<real_type, 7> const phi_values = {
            0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 1.0};
        inp.grid_phi.resize(phi_values.size());
        std::transform(phi_values.begin(),
                       phi_values.end(),
                       inp.grid_phi.begin(),
                       [](real_type phi) { return RealTurn{phi}; });
        inp.grid_z = {-150, -100, -50, 0, 50, 100, 150};

        // Initialize field values with a predominantly z-directed field
        size_type const nr = inp.grid_r.size();
        size_type const nphi = inp.grid_phi.size();
        size_type const nz = inp.grid_z.size();
        Array<size_type, 4> const dims{
            nr, nphi, nz, static_cast<size_type>(CylAxis::size_)};
        size_type const total_points = nr * nphi * nz;

        // Resize each component of the field
        inp.field.resize(static_cast<size_type>(CylAxis::size_) * total_points);

        // Fill with a simple field pattern
        HyperslabIndexer const flat_index{dims};
        for (size_type ir = 0; ir < nr; ++ir)
        {
            real_type r = inp.grid_r[ir];
            for (size_type iphi = 0; iphi < nphi; ++iphi)
            {
                // Convert turns to radians
                real_type phi = inp.grid_phi[iphi].value() * 2 * constants::pi;
                for (size_type iz = 0; iz < nz; ++iz)
                {
                    // Index calculation for 3D array
                    auto idx = flat_index(ir, iphi, iz, 0);

                    // Set field components
                    inp.field[idx + static_cast<size_type>(CylAxis::r)]
                        = 0.02 * r / 100 * std::cos(phi);
                    inp.field[idx + static_cast<size_type>(CylAxis::phi)]
                        = 0.02 * r / 100 * std::sin(phi);
                    inp.field[idx + static_cast<size_type>(CylAxis::z)]
                        = 3.8 - 0.0005 * (r / 100) * (r / 100);
                }
            }
        }
        return CylMapFieldParams(inp);
    }();

    CylMapField calc_field(field_map.host_ref());

    // Define samples in cylindrical coordinates
    size_type const nr_samples = 2;
    size_type const nphi_samples = 2;
    size_type const nz_samples = 2;

    // Define sampling ranges
    real_type r_min = 10;
    real_type r_max = 100;
    real_type phi_min = 0;
    real_type phi_max = constants::pi.value() / 2;
    real_type z_min = -100;
    real_type z_max = 100;

    std::vector<real_type> actual;

    for (size_type ir = 0; ir < nr_samples; ++ir)
    {
        real_type r = r_min + ir * (r_max - r_min) / (nr_samples - 1);
        for (size_type iphi = 0; iphi < nphi_samples; ++iphi)
        {
            real_type phi = phi_min
                            + iphi * (phi_max - phi_min) / (nphi_samples - 1);
            for (size_type iz = 0; iz < nz_samples; ++iz)
            {
                real_type z = z_min + iz * (z_max - z_min) / (nz_samples - 1);

                // Convert cylindrical to Cartesian coordinates for field
                // lookup
                Real3 pos{r * std::cos(phi), r * std::sin(phi), z};

                Real3 field = calc_field(pos);
                for (real_type f : field)
                {
                    actual.push_back(f);
                }
            }
        }
    }

    // Expected field values at the 8 sample points (2×2×2 grid in r, phi, z)
    // clang-format off
    static real_type const expected_field[] = {
        0.002,                0, 3.799975, // r=10cm,  phi=0,    z=-100cm
        0.002,                0, 3.799975, // r=10cm,  phi=0,    z=100cm
        -0.00173205080756888, 0, 3.799975, // r=10cm,  phi=pi/2, z=-100cm
        -0.00173205080756888, 0, 3.799975, // r=10cm,  phi=pi/2, z=100cm
        0.02,                 0, 3.7995,   // r=100cm, phi=0,    z=-100cm
        0.02,                 0, 3.7995,   // r=100cm, phi=0,    z=100cm
        -0.0173205062747002,  0, 3.7995,   // r=100cm, phi=pi/2, z=-100cm
        -0.0173205062747002,  0, 3.7995,   // r=100cm, phi=pi/2, z=100cm
    };
    // clang-format on

    EXPECT_VEC_NEAR(expected_field, actual, real_type{1e-7});
}

#if !CELERITAS_USE_COVFIE
#    define CartMapFieldTest DISABLED_CartMapFieldTest
#endif
using CartMapFieldTest = ::celeritas::test::Test;

CartMapFieldInput build_cart_map_input()
{
    CartMapFieldInput inp;
    inp.x.min = -2750;
    inp.x.max = 2750;
    inp.x.num = static_cast<size_type>(inp.x.max * 2 / 100);
    inp.y.min = -2750;
    inp.y.max = 2750;
    inp.y.num = static_cast<size_type>(inp.y.max * 2 / 100);
    inp.z.min = -6350;
    inp.z.max = 6350;
    inp.z.num = static_cast<size_type>(inp.z.max * 2 / 100);
    Array<size_type, 4> const dims{
        inp.x.num, inp.y.num, inp.z.num, static_cast<size_type>(Axis::size_)};
    size_type const total_points = inp.x.num * inp.y.num * inp.z.num;

    // Resize each component of the field
    inp.field.resize(static_cast<size_type>(Axis::size_) * total_points);

    // Fill with a simple field pattern
    HyperslabIndexer const flat_index{dims};
    for (size_type x = 0; x < inp.x.num; ++x)
    {
        for (size_type y = 0; y < inp.y.num; ++y)
        {
            for (size_type z = 0; z < inp.z.num; ++z)
            {
                auto arr = inp.field.begin() + flat_index(x, y, z, 0);
                arr[static_cast<size_type>(Axis::x)] = std::cos(x);
                arr[static_cast<size_type>(Axis::y)] = std::sin(y);
                arr[static_cast<size_type>(Axis::z)] = std::tan(z);
            }
        }
    }
    return inp;
}

TEST_F(CartMapFieldTest, host)
{
    CartMapFieldInput inp = build_cart_map_input();
    CartMapFieldParams field_map{inp};

    // FIXME: test data should be single-precision
    CartMapField calc_field(field_map.host_ref());

    // Sample the field

    // Define samples in cylindrical coordinates
    size_type const nx_samples = 3;
    size_type const ny_samples = 3;
    size_type const nz_samples = 3;
    std::vector<real_type> actual;
    Interpolator interp_x({0, inp.x.min}, {nx_samples - 1, inp.x.max});
    Interpolator interp_y({0, inp.y.min}, {ny_samples - 1, inp.y.max});
    Interpolator interp_z({0, inp.z.min}, {nz_samples - 1, inp.z.max});
    for (size_type ix = 0; ix < nx_samples; ++ix)
    {
        real_type x
            = std::min(interp_x(static_cast<real_type>(ix)), inp.x.max - 1);

        for (size_type iy = 0; iy < ny_samples; ++iy)
        {
            real_type y = std::min(interp_y(static_cast<real_type>(iy)),
                                   inp.y.max - 1);
            for (size_type iz = 0; iz < nz_samples; ++iz)
            {
                real_type z = std::min(interp_z(static_cast<real_type>(iz)),
                                       inp.z.max - 1);

                Real3 field = calc_field({x, y, z});
                for (real_type f : field)
                {
                    actual.push_back(f);
                }
            }
        }
    }

    // Check that the field values are as expected
    static real_type const expected_field[] = {
        1,
        0,
        0,
        1,
        0,
        0.169758066535,
        1,
        0,
        0.33834862709045,
        1,
        0.95637559890747,
        0,
        1,
        0.95637559890747,
        0.169758066535,
        1,
        0.95637553930283,
        0.33834862709045,
        1,
        -0.54941469430923,
        0,
        1,
        -0.54941469430923,
        0.169758066535,
        1,
        -0.54941469430923,
        0.33834865689278,
        -0.2921370267868,
        0,
        0,
        -0.2921370267868,
        0,
        0.169758066535,
        -0.2921370267868,
        0,
        0.33834862709045,
        -0.2921370267868,
        0.95637559890747,
        0,
        -0.2921370267868,
        0.95637559890747,
        0.169758066535,
        -0.2921370267868,
        0.95637559890747,
        0.33834865689278,
        -0.2921370267868,
        -0.54941469430923,
        0,
        -0.2921370267868,
        -0.54941469430923,
        0.16975805163383,
        -0.2921370267868,
        -0.54941469430923,
        0.33834862709045,
        -0.83018344640732,
        0,
        0,
        -0.83018350601196,
        0,
        0.169758066535,
        -0.83018344640732,
        0,
        0.33834865689278,
        -0.83018344640732,
        0.95637553930283,
        0,
        -0.83018350601196,
        0.95637559890747,
        0.16975805163383,
        -0.83018344640732,
        0.95637553930283,
        0.33834862709045,
        -0.83018344640732,
        -0.54941469430923,
        0,
        -0.83018344640732,
        -0.54941469430923,
        0.16975805163383,
        -0.83018350601196,
        -0.54941475391388,
        0.33834865689278,
    };
    EXPECT_VEC_NEAR(expected_field, actual, real_type{1e-6});
}

TEST_F(CartMapFieldTest, TEST_IF_CELER_DEVICE(device))
{
    Array<size_type, 3> n_samples{3, 3, 3};

    // FIXME: these should be single-precision for covfie
    std::vector<real_type> field_values(n_samples[0] * n_samples[1]
                                        * n_samples[2] * 3);

    auto input = build_cart_map_input();
    auto span = make_span(field_values);

    // Run the test on device
    field_test(input, span, n_samples);

    static real_type const expected_field[]
        = {1,         0,         0,        1,         0,         0.16975,
           1,         0,         0.336311, 1,         0.956376,  0,
           1,         0.956376,  0.16975,  1,         0.956376,  0.336311,
           1,         -0.547601, 0,        1,         -0.547601, 0.16975,
           1,         -0.547601, 0.336311, -0.292139, 0,         0,
           -0.292139, 0,         0.16975,  -0.292139, 0,         0.336311,
           -0.292139, 0.956376,  0,        -0.292139, 0.956376,  0.16975,
           -0.292139, 0.956376,  0.336311, -0.292139, -0.547601, 0,
           -0.292139, -0.547601, 0.16975,  -0.292139, -0.547601, 0.336311,
           -0.830352, 0,         0,        -0.830352, 0,         0.16975,
           -0.830352, 0,         0.336311, -0.830352, 0.956376,  0,
           -0.830352, 0.956376,  0.16975,  -0.830352, 0.956376,  0.336311,
           -0.830352, -0.547601, 0,        -0.830352, -0.547601, 0.16975,
           -0.830352, -0.547601, 0.336311};

    // FIXME: reference values use lower-precision texture interpolation
    constexpr real_type tol = CELERITAS_USE_HIP ? 1e-2 : 1e-5;
    EXPECT_VEC_NEAR(expected_field, field_values, tol);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
