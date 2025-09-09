//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/MatrixUtils.test.cc
//---------------------------------------------------------------------------//
#include "orange/MatrixUtils.hh"

#include <cmath>

#include "corecel/math/ArrayUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class MatrixUtilsTest : public Test
{
  public:
    template<class T, size_type N>
    std::vector<T> flattened(SquareMatrix<T, N> const& inp) const
    {
        std::vector<T> result;

        for (auto& row : inp)
        {
            result.insert(result.end(), row.begin(), row.end());
        }
        return result;
    }
};

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, gemv)
{
    // clang-format off
    using Vec3 = Array<int, 3>;
    using Mat3 = SquareMatrix<int, 3>;
    Mat3 const mat{
        Vec3{1, 2, 3},
        Vec3{0, 1, 2},
        Vec3{4, 0, 1}
    };
    // clang-format on

    Vec3 result = gemv(10, mat, {6, 7, 8}, 2, {-1, -2, -3});
    EXPECT_EQ((2 * -1) + 10 * (6 * 1 + 7 * 2 + 8 * 3), result[0]);
    EXPECT_EQ((2 * -2) + 10 * (6 * 0 + 7 * 1 + 8 * 2), result[1]);
    EXPECT_EQ((2 * -3) + 10 * (6 * 4 + 7 * 0 + 8 * 1), result[2]);

    result = gemv(matrix::transpose, 10, mat, {6, 7, 8}, 2, {-1, -2, -3});
    EXPECT_EQ((2 * -1) + 10 * (6 * 1 + 7 * 0 + 8 * 4), result[0]);
    EXPECT_EQ((2 * -2) + 10 * (6 * 2 + 7 * 1 + 8 * 0), result[1]);
    EXPECT_EQ((2 * -3) + 10 * (6 * 3 + 7 * 2 + 8 * 1), result[2]);

    result = gemv(mat, {6, 7, 8});
    EXPECT_EQ((6 * 1 + 7 * 2 + 8 * 3), result[0]);
    EXPECT_EQ((6 * 0 + 7 * 1 + 8 * 2), result[1]);
    EXPECT_EQ((6 * 4 + 7 * 0 + 8 * 1), result[2]);

    result = gemv(matrix::transpose, mat, {6, 7, 8});
    EXPECT_EQ((6 * 1 + 7 * 0 + 8 * 4), result[0]);
    EXPECT_EQ((6 * 2 + 7 * 1 + 8 * 0), result[1]);
    EXPECT_EQ((6 * 3 + 7 * 2 + 8 * 1), result[2]);
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, determinant)
{
    using Vec3 = Array<int, 3>;
    using Mat3 = SquareMatrix<int, 3>;
    Mat3 const a{Vec3{1, 2, 3}, Vec3{-1, 0, 1}, Vec3{-3, 2, -1}};

    EXPECT_EQ(-16, determinant(a));
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, trace)
{
    using Vec3 = Array<int, 3>;
    using Mat3 = SquareMatrix<int, 3>;
    Mat3 const a{Vec3{1, 2, 3}, Vec3{-1, 4, 1}, Vec3{-3, 2, -1}};

    EXPECT_EQ(4, trace(a));
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, gemm)
{
    using Vec3 = Array<int, 3>;
    using Mat3 = SquareMatrix<int, 3>;
    Mat3 const a{Vec3{1, 2, 3}, Vec3{-1, 0, 1}, Vec3{-3, 2, -1}};
    Mat3 const b{Vec3{2, 1, 1}, Vec3{-1, 2, 1}, Vec3{3, 1, -1}};

    auto result = gemm(a, b);
    EXPECT_EQ(result, (Mat3{Vec3{9, 8, 0}, Vec3{1, 0, -2}, Vec3{-11, 0, 0}}));

    result = gemm(matrix::transpose, a, b);
    EXPECT_EQ(result, (Mat3{Vec3{-6, -4, 3}, Vec3{10, 4, 0}, Vec3{2, 4, 5}}));
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, orthonormalize)
{
    using Vec3 = Array<double, 3>;
    using Mat3 = SquareMatrix<double, 3>;
    Mat3 const a{Vec3{1, 2, 3}, Vec3{-1, 0, 1}, Vec3{-3, 2, -1}};

    Mat3 result{a};
    orthonormalize(&result);
    EXPECT_SOFT_EQ(1, std::fabs(determinant(result)));

    static double const expected[] = {0.26726124191242,
                                      0.53452248382485,
                                      0.80178372573727,
                                      -0.87287156094397,
                                      -0.21821789023599,
                                      0.43643578047198,
                                      -0.40824829046386,
                                      0.81649658092773,
                                      -0.40824829046386};
    EXPECT_VEC_SOFT_EQ(expected, flattened(result));

    if (CELERITAS_DEBUG)
    {
        // Make singular matrix
        Mat3 result{a};
        result[2] = result[1];

        EXPECT_THROW(orthonormalize(&result), DebugError);
    }
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, make_rotation)
{
    {
        SCOPED_TRACE("identity");
        for (auto ax : range(Axis::size_))
        {
            auto r = make_rotation(ax, Turn{0.});
            static double const expected_flattened[]
                = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            EXPECT_VEC_EQ(expected_flattened, flattened(r));
        }
    }
    {
        SCOPED_TRACE("x");
        auto r = make_rotation(Axis::x, native_value_to<Turn>(std::acos(0.3)));
        EXPECT_SOFT_EQ(2 * 0.3 + 1, trace(r));
        static double const expected_r[]
            = {1, 0, 0, 0, 0.3, -0.95393920141695, 0, 0.95393920141695, 0.3};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));

        static double const expected_rotated[]
            = {1, -2.2618176042508, 2.8078784028339};
        EXPECT_VEC_SOFT_EQ(expected_rotated, gemv(r, {1, 2, 3}));
    }
    {
        SCOPED_TRACE("y");
        auto r = make_rotation(Axis::y, Turn{-0.125});  // -45 degrees
        static double const expected_r[] = {0.70710678118655,
                                            0,
                                            -0.70710678118655,
                                            0,
                                            1,
                                            0,
                                            0.70710678118655,
                                            0,
                                            0.70710678118655};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));

        static double const expected_rotated[]
            = {-1.4142135623731, 2, 2.8284271247462};
        EXPECT_VEC_SOFT_EQ(expected_rotated, gemv(r, {1, 2, 3}));
    }
    {
        SCOPED_TRACE("z");
        auto r = make_rotation(Axis::z, Turn{0.9});  // 324 degrees
        EXPECT_SOFT_EQ(2 * cos(Turn{0.9}) + 1, trace(r));
        static double const expected_r[] = {0.80901699437495,
                                            0.58778525229247,
                                            0,
                                            -0.58778525229247,
                                            0.80901699437495,
                                            0,
                                            0,
                                            0,
                                            1};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));

        static double const expected_rotated[]
            = {1.9845874989599, 1.0302487364574, 3};
        EXPECT_VEC_SOFT_EQ(expected_rotated, gemv(r, {1, 2, 3}));
    }
    {
        SCOPED_TRACE("x exact");
        auto r = make_rotation(Axis::x, Turn{0.5});
        static double const expected_r[] = {1, 0, 0, 0, -1, 0, 0, 0, -1};
        EXPECT_VEC_EQ(expected_r, flattened(r));

        static double const expected_rotated[] = {1, -2, -3};
        EXPECT_VEC_EQ(expected_rotated, gemv(r, {1, 2, 3}));
    }
    {
        SCOPED_TRACE("y exact");
        auto r = make_rotation(Axis::y, Turn{-0.25});
        static double const expected_r[] = {0, 0, -1, 0, 1, 0, 1, 0, 0};
        EXPECT_VEC_EQ(expected_r, flattened(r));

        static double const expected_rotated[] = {-3, 2, 1};
        EXPECT_VEC_EQ(expected_rotated, gemv(r, {1, 2, 3}));
    }
    {
        SCOPED_TRACE("z exact");
        auto r = make_rotation(Axis::z, Turn{1.75});
        static double const expected_r[] = {0, 1, 0, -1, 0, 0, 0, 0, 1};
        EXPECT_VEC_EQ(expected_r, flattened(r));

        static double const expected_rotated[] = {2, -1, 3};
        EXPECT_VEC_EQ(expected_rotated, gemv(r, {1, 2, 3}));
    }
    {
        SCOPED_TRACE("z quarter, x quarter");
        auto r = make_rotation(
            Axis::x, Turn{0.25}, make_rotation(Axis::z, Turn{0.25}));
        static double const expected_r[] = {0, -1, 0, 0, 0, -1, 1, 0, 0};
        EXPECT_VEC_EQ(expected_r, flattened(r));

        static double const expected_rotated[] = {-2, -3, 1};
        EXPECT_VEC_EQ(expected_rotated, gemv(r, {1, 2, 3}));
    }
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, make_arb_rotation)
{
    {
        auto turn = native_value_to<Turn>(std::acos(0.3));
        EXPECT_VEC_SOFT_EQ(flattened(make_rotation(Axis::x, turn)),
                           flattened(make_rotation({1, 0, 0}, turn)));
    }
    {
        auto r = make_rotation(make_unit_vector(Real3{1, 1, 1}), Turn{0.25});
        static double const expected_r[] = {
            0.33333333333333,
            -0.24401693585629,
            0.91068360252296,
            0.91068360252296,
            0.33333333333333,
            -0.24401693585629,
            -0.24401693585629,
            0.91068360252296,
            0.33333333333333,
        };
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));
    }
    {
        auto r = make_rotation(make_unit_vector(Real3{1, 0, 1}), Turn{0.5});

        static double const expected_r[] = {0, 0, 1, 0, -1, 0, 1, 0, 0};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));
    }
    {
        auto r = make_rotation(make_unit_vector(Real3{-3, -4, -5}),
                               Turn{1 / real_type{6}});
        static double const expected_r[] = {
            0.59,
            0.73237243569579,
            -0.33989794855664,
            -0.49237243569579,
            0.66,
            0.56742346141748,
            0.63989794855664,
            -0.16742346141748,
            0.75,
        };
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));
    }
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, make_scaling)
{
    {
        auto r = make_scaling(real_type{2.0});

        static double const expected_r[] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));
    }
    {
        auto r = make_scaling(Axis::z, real_type{2.0});

        static double const expected_r[] = {1, 0, 0, 0, 1, 0, 0, 0, 2};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));
    }
    {
        auto r = make_scaling({2, 3, 4});

        static double const expected_r[] = {2, 0, 0, 0, 3, 0, 0, 0, 4};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));
    }
}

//---------------------------------------------------------------------------//

TEST_F(MatrixUtilsTest, make_reflection)
{
    {
        auto r = make_reflection(Axis::z);

        static double const expected_r[] = {1, 0, 0, 0, 1, 0, 0, 0, -1};
        EXPECT_VEC_SOFT_EQ(expected_r, flattened(r));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
