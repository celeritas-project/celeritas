//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/detail/SurfaceGridHash.test.cc
//---------------------------------------------------------------------------//
#include "orange/orangeinp/detail/SurfaceGridHash.hh"

#include <algorithm>

#include "celeritas_test.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
using VecInt = std::vector<int>;

//! Apply the functor to values of all elements with the same key
template<class F>
void apply_equal_range(SurfaceGridHash const& sgh,
                       SurfaceGridHash::const_iterator iter,
                       F&& func)
{
    // TODO: we can't use reverse iterators and we can't search based on the
    // iterator's bucket... this is going to lose some efficiency by re-hashing
    // the key and doing another search.
    auto&& [first, last] = sgh.equal_range(iter);
    for (; first != last; ++first)
    {
        func(first->second);
    }
}

//! Get a sorted list of surfaces that share the same bin
VecInt get_surfaces(SurfaceGridHash const& sgh, SurfaceGridHash::iterator iter)
{
    std::vector<int> result;
    apply_equal_range(sgh, iter, [&result](LocalSurfaceId lsid) {
        CELER_EXPECT(lsid);
        result.push_back(lsid.get());
    });
    std::sort(result.begin(), result.end());
    return result;
}

//---------------------------------------------------------------------------//
class SurfaceGridHashTest : public ::celeritas::test::Test
{
  protected:
    void SetUp() override {}
};

TEST_F(SurfaceGridHashTest, insertion)
{
    real_type const grid_size{0.1};
    real_type const tol{1e-4};
    SurfaceGridHash grid{grid_size, tol};

    real_type const grid_offset{grid_size / 2};
    LocalSurfaceId next_id{0};

    {
        // Insert at the middle of the bin, where there shouldn't be collisions
        auto&& [first, second] = grid.insert(SurfaceType::px, 0.0, next_id++);
        EXPECT_NE(first, second);
        EXPECT_EQ(second, grid.end());
        EXPECT_VEC_EQ((VecInt{0}), get_surfaces(grid, first));
        EXPECT_VEC_EQ((VecInt{}), get_surfaces(grid, second));
    }
    {
        // Insert near the end of the bin, where we should get two iterators
        auto&& [first, second]
            = grid.insert(SurfaceType::px, grid_offset - 0.5 * tol, next_id++);
        EXPECT_NE(first, second);
        EXPECT_NE(second, grid.end());
        EXPECT_VEC_EQ((VecInt{0, 1}), get_surfaces(grid, first));
        EXPECT_VEC_EQ((VecInt{1}), get_surfaces(grid, second));
    }
    {
        // Insert near the end of the previous bin
        auto&& [first, second] = grid.insert(
            SurfaceType::px, -grid_offset - 0.5 * tol, next_id++);
        EXPECT_NE(first, second);
        EXPECT_NE(second, grid.end());
        EXPECT_VEC_EQ((VecInt{2}), get_surfaces(grid, first));
        EXPECT_VEC_EQ((VecInt{0, 1, 2}), get_surfaces(grid, second));
    }
    {
        // Insert in the middle of the right bin
        auto&& [first, second]
            = grid.insert(SurfaceType::px, grid_size, next_id++);
        EXPECT_NE(first, second);
        EXPECT_EQ(second, grid.end());
        EXPECT_VEC_EQ((VecInt{1, 3}), get_surfaces(grid, first));
        EXPECT_VEC_EQ((VecInt{}), get_surfaces(grid, second));
    }
    {
        // Insert a surface of a different type but the same hash point
        auto&& [first, second] = grid.insert(SurfaceType::py, 0.0, next_id++);
        EXPECT_NE(first, second);
        EXPECT_EQ(second, grid.end());
        EXPECT_VEC_EQ((VecInt{4}), get_surfaces(grid, first));
        EXPECT_VEC_EQ((VecInt{}), get_surfaces(grid, second));
    }
    {
        // Insert yet another surface near the end grid point
        auto&& [first, second]
            = grid.insert(SurfaceType::pz, grid_offset - 0.5 * tol, next_id++);
        EXPECT_NE(first, second);
        EXPECT_NE(second, grid.end());
        EXPECT_VEC_EQ((VecInt{5}), get_surfaces(grid, first));
        EXPECT_VEC_EQ((VecInt{5}), get_surfaces(grid, second));
    }
}

TEST_F(SurfaceGridHashTest, deleting)
{
    real_type const grid_size{10};
    real_type const tol{0.1};
    SurfaceGridHash grid{grid_size, tol};

    LocalSurfaceId next_id{0};

    {
        grid.insert(SurfaceType::px, 0.0, next_id++);
        grid.insert(SurfaceType::px, 5.0, next_id++);
        grid.insert(SurfaceType::px, 20.0, next_id++);
        grid.insert(SurfaceType::px, 40.0, next_id++);
    }
    {
        // Add and erase
        auto&& [first, second] = grid.insert(SurfaceType::pz, 0, next_id++);
        grid.erase(first);
        if (CELERITAS_DEBUG)
        {
            EXPECT_THROW(grid.erase(second), DebugError);
        }
    }
    {
        // Add and erase
        auto&& [first, second] = grid.insert(SurfaceType::px, 0, next_id++);
        EXPECT_VEC_EQ((VecInt{0, 1, 5}), get_surfaces(grid, first));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace orangeinp
}  // namespace celeritas
