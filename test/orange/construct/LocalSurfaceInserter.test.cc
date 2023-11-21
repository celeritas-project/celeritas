//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/construct/LocalSurfaceInserter.test.cc
//---------------------------------------------------------------------------//
#include "orange/construct/detail/LocalSurfaceInserter.hh"

#include <random>

#include "corecel/sys/Stopwatch.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//

class LocalSurfaceInserterTest : public ::celeritas::test::Test
{
  protected:
    static constexpr real_type small = 1e-5;  // small < eps
    static constexpr real_type eps = 1e-4;
    static constexpr real_type large = 1e-3;  // eps < large < sqrt(eps)
    SoftSurfaceEqual softeq_{eps};

    void SetUp() override
    {
        tol = Tolerance<>::from_relative(eps);
        CELER_ENSURE(tol);
    }

    LocalSurfaceInserter::VecSurface surfaces;
    Tolerance<> tol;
};

TEST_F(LocalSurfaceInserterTest, no_duplicates)
{
    LocalSurfaceInserter insert(&surfaces, tol);

    EXPECT_EQ(0, insert(PlaneX{2.0}).unchecked_get());
    EXPECT_EQ(1, insert(PlaneY{2.0}).unchecked_get());
    EXPECT_EQ(2, insert(PlaneZ{2.0}).unchecked_get());
    EXPECT_EQ(3, surfaces.size());
}

TEST_F(LocalSurfaceInserterTest, exact_duplicates)
{
    LocalSurfaceInserter insert(&surfaces, tol);

    for (int i = 0; i < 3; ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0, insert(PlaneX{2.0}).unchecked_get());
        EXPECT_EQ(1, insert(PlaneY{2.0}).unchecked_get());
    }
    EXPECT_EQ(2, surfaces.size());
}

/*!
 * Insert surfaces that are very close to each other. Because we keep the
 * deduplicated but *not exactly equal* surfaces, the vector size grows.
 */
TEST_F(LocalSurfaceInserterTest, tiny_duplicates)
{
    LocalSurfaceInserter insert(&surfaces, tol);

    for (int i = 0; i < 3; ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0, insert(PlaneX{2 + small * i}).unchecked_get());
        EXPECT_EQ(1, insert(PlaneY{2 + small * i}).unchecked_get());
    }

    EXPECT_EQ(6, insert(PlaneZ{2}).unchecked_get());
    EXPECT_EQ(7, surfaces.size());
}

/*!
 * Insert surfaces that each have a gap of less than epsilon, but the
 * first and third have a combined gap of *more*. This means insertion order
 * changes the result, and could cause particles to be "lost" (need more than
 * one bump) if jumping into a lower level.
 */
TEST_F(LocalSurfaceInserterTest, chained_duplicates)
{
    LocalSurfaceInserter insert(&surfaces, tol);

    EXPECT_EQ(0, insert(PlaneX{2}).unchecked_get());
    EXPECT_EQ(1, insert(PlaneY{2}).unchecked_get());

    for (int i = 1; i < 4; ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0, insert(PlaneX{2 + i * eps / 2}).unchecked_get());
    }
    EXPECT_EQ(5, surfaces.size());
}

TEST_F(LocalSurfaceInserterTest, DISABLED_performance_test)
{
    std::mt19937 rng;
    UniformRealDistribution<> sample_point{-1, 1};
    UniformBoxDistribution<> sample_box{{-1, -1, -1}, {1, 1, 1}};

    for (int num_samples = 16; num_samples < 40000; num_samples *= 2)
    {
        cout << "Sampling " << num_samples << "..." << std::flush;
        surfaces.reserve(num_samples * 2);
        surfaces.clear();
        LocalSurfaceInserter insert(&surfaces, tol);

        Stopwatch get_time;
        for (int i = 0; i < num_samples; ++i)
        {
            insert(Sphere(sample_box(rng), 1.0));
            insert(PlaneX(sample_point(rng)));
        }
        cout << get_time() << " s" << endl;
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
