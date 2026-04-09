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
#include "celeritas/field/LoadCovfieField.hh"
#include "celeritas/field/RZMapField.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/RZMapFieldParams.hh"
#include "celeritas/field/UniformField.hh"
#include "celeritas/field/UniformZField.hh"

#include "CMSParameterizedField.hh"
#include "CovfieTestField.hh"
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
            actual.push_back(native_value_to<units::TeslaField>(f).value());
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

#if !CELERITAS_USE_COVFIE
#    define RZMapFieldTest DISABLED_RZMapFieldTest
#endif
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
            actual.push_back(native_value_to<units::TeslaField>(f).value());
        }
    }

    // clang-format off
    // Covfie bilinear interpolation on the RZ grid (both Br and Bz
    // interpolated in 2D)
    static real_type const expected_field[] = {
        -0,                  -0,                  3.811202287674,
        0.000557730426456419, 0.000557730426456419, 3.8078203125,
        0.0023259673673599,  0.0023259673673599,  3.804498046875,
        0.0053047110587328,  0.0053047110587328,  3.8012359375,
        0.0094939613342285,  0.0094939613342285,  3.7980328125,
        0.0149389093193622,  0.0149389093193622,  3.7849625,
        0.0215377738208854,  0.0215377738208854,  3.7723875,
        0.0283599192434375,  0.0283599192434375,  3.762944140625,
    };
    // clang-format on
    // Float32 covfie interpolation; allow for AVX2 rounding variation
    EXPECT_VEC_NEAR(expected_field, actual, real_type{2e-7});
}

TEST_F(RZMapFieldTest, interp_validation)
{
    // Validate bilinear interpolation with a synthetic field: Br(r,z) = r + z,
    // Bz(r,z) = r + z (each component varies on both axes). True bilinear
    // interpolation reproduces linear functions exactly.
    //
    // Use a 2x2 grid so the midpoint r=1, z=1 has frac=(0.5, 0.5) exactly.
    // All values are in native units (no unit conversion).
    {
        RZMapFieldInput inp;
        inp.num_grid_r = 2;
        inp.num_grid_z = 2;
        inp.min_r = 0;
        inp.max_r = 2;
        inp.min_z = 0;
        inp.max_z = 2;
        inp.field_r.resize(4);
        inp.field_z.resize(4);
        for (int iz = 0; iz < 2; ++iz)
            for (int ir = 0; ir < 2; ++ir)
            {
                double r = ir * 2.0, z = iz * 2.0;
                inp.field_r[iz * 2 + ir] = r + z;  // Br = r + z
                inp.field_z[iz * 2 + ir] = r + z;  // Bz = r + z
            }

        RZMapFieldParams params(inp);
        RZMapField field(params.host_ref());

        // At midpoint r=1, z=1: Br = Bz = 2 (exact bilinear).
        // Bx = Br*(x/r) = 2*(1/1) = 2, By = 0, Bz = 2.
        auto result = field(Real3{real_type(1), 0, real_type(1)});
        EXPECT_SOFT_EQ(2.0, result[0]);
        EXPECT_SOFT_EQ(0.0, result[1]);
        EXPECT_SOFT_EQ(2.0, result[2]);
    }
}

TEST_F(RZMapFieldTest, TEST_IF_CELER_DEVICE(device))
{
    RZMapFieldInput inp;
    {
        auto filename
            = this->test_data_path("celeritas", "cms-tiny.field.json");
        std::ifstream(filename) >> inp;
    }

    // Build sample points: same as host test
    int const nsamples = 8;
    real_type delta_z = from_cm(25.0);
    real_type delta_r = from_cm(12.0);

    std::vector<Real3> points(nsamples);
    for (int i : range(nsamples))
    {
        points[i] = {i * delta_r, i * delta_r, i * delta_z};
    }

    std::vector<real_type> field_values(nsamples * 3);

    // Run the test on device
    rzfield_test(inp,
                 Span<Real3 const>{points.data(), points.size()},
                 Span<real_type>{field_values.data(), field_values.size()});

    // Convert to Tesla for comparison
    std::vector<real_type> actual;
    for (auto f : field_values)
    {
        actual.push_back(native_value_to<units::TeslaField>(f).value());
    }

    // Same expected values as host: device uses software interpolation
    // (no CUDA texture), so results are identical.
    // clang-format off
    static real_type const expected_field[] = {
        -0,                  -0,                  3.811202287674,
        0.000557730426456419, 0.000557730426456419, 3.8078203125,
        0.0023259673673599,  0.0023259673673599,  3.804498046875,
        0.0053047110587328,  0.0053047110587328,  3.8012359375,
        0.0094939613342285,  0.0094939613342285,  3.7980328125,
        0.0149389093193622,  0.0149389093193622,  3.7849625,
        0.0215377738208854,  0.0215377738208854,  3.7723875,
        0.0283599192434375,  0.0283599192434375,  3.762944140625,
    };
    // clang-format on
    EXPECT_VEC_NEAR(expected_field, actual, real_type{2e-7});
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

    // Expected field values at the 8 sample points (2x2x2 grid in r, phi, z)
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

//---------------------------------------------------------------------------//
// COVFIE IMPORT TESTS
//---------------------------------------------------------------------------//

#if !CELERITAS_USE_COVFIE
#    define CovfieCartImportTest DISABLED_CovfieCartImportTest
#    define CovfieRZImportTest DISABLED_CovfieRZImportTest
#endif

using CovfieCartImportTest = Test;
using CovfieRZImportTest = Test;

TEST_F(CovfieCartImportTest, load_2x3x4)
{
    auto path = this->make_unique_filename(".covfie");
    write_cart_covfie(path, 2, 3, 4);

    auto inp = load_covfie_cart_field(path);

    // Check grid dimensions
    EXPECT_EQ(2u, inp.x.num);
    EXPECT_EQ(3u, inp.y.num);
    EXPECT_EQ(4u, inp.z.num);

    // Check grid bounds (identity affine: [0, n-1])
    EXPECT_SOFT_EQ(0.0, inp.x.min);
    EXPECT_SOFT_EQ(1.0, inp.x.max);
    EXPECT_SOFT_EQ(0.0, inp.y.min);
    EXPECT_SOFT_EQ(2.0, inp.y.max);
    EXPECT_SOFT_EQ(0.0, inp.z.min);
    EXPECT_SOFT_EQ(3.0, inp.z.max);

    // Check total field size: 3 * 2 * 3 * 4 = 72
    EXPECT_EQ(72u, inp.field.size());

    // Check field values at corners to verify stride ordering [X][Y][Z][3]
    // At (ix=0, iy=0, iz=0): base = 0, expect (0, 0, 0)
    EXPECT_SOFT_EQ(0.0, inp.field[0]);
    EXPECT_SOFT_EQ(0.0, inp.field[1]);
    EXPECT_SOFT_EQ(0.0, inp.field[2]);

    // At (ix=1, iy=0, iz=0): base = (1*3 + 0)*4 + 0 = 12, offset*3 = 36
    EXPECT_SOFT_EQ(1.0, inp.field[36]);  // Bx = ix = 1
    EXPECT_SOFT_EQ(0.0, inp.field[37]);  // By = iy = 0
    EXPECT_SOFT_EQ(0.0, inp.field[38]);  // Bz = iz = 0

    // At (ix=0, iy=2, iz=3): base = (0*3 + 2)*4 + 3 = 11, offset*3 = 33
    EXPECT_SOFT_EQ(0.0, inp.field[33]);  // Bx = 0
    EXPECT_SOFT_EQ(2.0, inp.field[34]);  // By = 2
    EXPECT_SOFT_EQ(3.0, inp.field[35]);  // Bz = 3

    // At (ix=1, iy=2, iz=3): base = (1*3 + 2)*4 + 3 = 23, offset*3 = 69
    EXPECT_SOFT_EQ(1.0, inp.field[69]);  // Bx = 1
    EXPECT_SOFT_EQ(2.0, inp.field[70]);  // By = 2
    EXPECT_SOFT_EQ(3.0, inp.field[71]);  // Bz = 3

    // Check validity
    EXPECT_TRUE(static_cast<bool>(inp));
}

TEST_F(CovfieRZImportTest, load_2x3)
{
    auto path = this->make_unique_filename(".covfie");
    write_rz_covfie(path, 2, 3);

    auto inp = load_covfie_rz_field(path);

    // Check grid dimensions
    EXPECT_EQ(2u, inp.num_grid_r);
    EXPECT_EQ(3u, inp.num_grid_z);

    // Check grid bounds (identity affine: r=[0,1], z=[0,2])
    EXPECT_SOFT_EQ(0.0, inp.min_r);
    EXPECT_SOFT_EQ(1.0, inp.max_r);
    EXPECT_SOFT_EQ(0.0, inp.min_z);
    EXPECT_SOFT_EQ(2.0, inp.max_z);

    // Check total field size: 2 * 3 = 6
    EXPECT_EQ(6u, inp.field_r.size());
    EXPECT_EQ(6u, inp.field_z.size());

    // Check field values with [Z][R] indexing (R stride 1)
    // At (ir=0, iz=0): idx = 0*2 + 0 = 0
    EXPECT_SOFT_EQ(0.0, inp.field_r[0]);  // Br = ir = 0
    EXPECT_SOFT_EQ(0.0, inp.field_z[0]);  // Bz = iz = 0

    // At (ir=1, iz=0): idx = 0*2 + 1 = 1
    EXPECT_SOFT_EQ(1.0, inp.field_r[1]);  // Br = ir = 1
    EXPECT_SOFT_EQ(0.0, inp.field_z[1]);  // Bz = iz = 0

    // At (ir=0, iz=2): idx = 2*2 + 0 = 4
    EXPECT_SOFT_EQ(0.0, inp.field_r[4]);  // Br = ir = 0
    EXPECT_SOFT_EQ(2.0, inp.field_z[4]);  // Bz = iz = 2

    // At (ir=1, iz=2): idx = 2*2 + 1 = 5
    EXPECT_SOFT_EQ(1.0, inp.field_r[5]);  // Br = ir = 1
    EXPECT_SOFT_EQ(2.0, inp.field_z[5]);  // Bz = iz = 2

    // Check validity
    EXPECT_TRUE(static_cast<bool>(inp));
}

//---------------------------------------------------------------------------//
// CART MAP FIELD TESTS (covfie-only)
//---------------------------------------------------------------------------//

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
