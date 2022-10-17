//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitIndexer.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/UnitIndexer.hh"

#include "celeritas/Types.hh"

#include "celeritas_test.hh"

using UnitIndexer = celeritas::detail::UnitIndexer;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

TEST(UnitIndexerTest, single)
{
    UnitIndexer indexer({4}, {10});

    EXPECT_EQ(1, indexer.num_universes());
    EXPECT_EQ(4, indexer.num_surfaces());
    EXPECT_EQ(10, indexer.num_volumes());

    EXPECT_EQ(SurfaceId(0),
              indexer.global_surface(UniverseId{0}, SurfaceId{0}));
    EXPECT_EQ(SurfaceId(3),
              indexer.global_surface(UniverseId{0}, SurfaceId{3}));

    EXPECT_EQ(VolumeId(0), indexer.global_volume(UniverseId{0}, VolumeId{0}));
    EXPECT_EQ(VolumeId(9), indexer.global_volume(UniverseId{0}, VolumeId{9}));

    UniverseId univ;
    SurfaceId  surf;
    std::tie(univ, surf) = indexer.local_surface(SurfaceId{0});
    EXPECT_EQ(UniverseId(0), univ);
    EXPECT_EQ(SurfaceId(0), surf);
    std::tie(univ, surf) = indexer.local_surface(SurfaceId{3});
    EXPECT_EQ(UniverseId(0), univ);
    EXPECT_EQ(SurfaceId(3), surf);

    VolumeId cell;
    std::tie(univ, cell) = indexer.local_volume(VolumeId{0});
    EXPECT_EQ(UniverseId(0), univ);
    EXPECT_EQ(VolumeId(0), cell);
    std::tie(univ, cell) = indexer.local_volume(VolumeId{3});
    EXPECT_EQ(UniverseId(0), univ);
    EXPECT_EQ(VolumeId(3), cell);

    EXPECT_THROW(indexer.global_surface(UniverseId{0}, SurfaceId{4}),
                 DebugError);
    EXPECT_THROW(indexer.global_surface(UniverseId{1}, SurfaceId{0}),
                 DebugError);
    EXPECT_THROW(indexer.global_volume(UniverseId{0}, VolumeId{10}),
                 DebugError);
    EXPECT_THROW(indexer.global_volume(UniverseId{1}, VolumeId{0}), DebugError);

    EXPECT_THROW(indexer.local_surface(SurfaceId(4)), DebugError);
    EXPECT_THROW(indexer.local_volume(VolumeId(10)), DebugError);
}

TEST(UniverseIndexerTest, multi)
{
    // One universe has zero surfaces
    const UnitIndexer::VecSize surfaces_per_uni{4, 1, 0, 1};
    const UnitIndexer::VecSize cells_per_uni{1, 1, 1, 2};
    UnitIndexer                indexer(surfaces_per_uni, cells_per_uni);
    EXPECT_EQ(6, indexer.num_surfaces());
    EXPECT_EQ(5, indexer.num_volumes());
    EXPECT_EQ(4, indexer.num_universes());

    unsigned int global_surface = 0;
    for (auto u : range(surfaces_per_uni.size()))
    {
        for (auto s : range(surfaces_per_uni[u]))
        {
            UniverseId univ;
            SurfaceId  surf;
            std::tie(univ, surf)
                = indexer.local_surface(SurfaceId{global_surface});
            EXPECT_EQ(u, univ.unchecked_get());
            EXPECT_EQ(s, surf.unchecked_get());
            EXPECT_EQ(global_surface,
                      indexer.global_surface(univ, surf).unchecked_get());
            ++global_surface;
        }
    }

    unsigned int global_volume = 0;
    for (auto u : range(cells_per_uni.size()))
    {
        for (auto c : range(cells_per_uni[u]))
        {
            UniverseId univ;
            VolumeId   cell;
            std::tie(univ, cell)
                = indexer.local_volume(VolumeId{global_volume});
            EXPECT_EQ(u, univ.unchecked_get());
            EXPECT_EQ(c, cell.unchecked_get());
            EXPECT_EQ(global_volume,
                      indexer.global_volume(univ, cell).unchecked_get());
            ++global_volume;
        }
    }
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
