//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UnitIndexer.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/UnitIndexer.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "celeritas/Types.hh"

#include "celeritas_test.hh"

using UnitIndexer = celeritas::detail::UnitIndexer;

namespace celeritas
{
namespace test
{
class UnitIndexerTest : public Test
{
  public:
    using CollectionHostRef
        = UnitIndexerData<Ownership::const_reference, MemSpace::host>;
    using VecSize = std::vector<size_type>;

    const CollectionHostRef& host_ref() const { return data_.host(); }

    void set_data(VecSize surfaces, VecSize volumes)
    {
        UnitIndexerData<Ownership::value, MemSpace::host> data;

        auto cb_s = make_builder(&data.surfaces);
        cb_s.insert_back(surfaces.begin(), surfaces.end());

        auto cb_v = make_builder(&data.volumes);
        cb_v.insert_back(volumes.begin(), volumes.end());

        data_ = CollectionMirror<UnitIndexerData>{std::move(data)};
    }

  protected:
    CollectionMirror<UnitIndexerData> data_;
};

//---------------------------------------------------------------------------//

TEST_F(UnitIndexerTest, single)
{
    set_data({0, 4}, {0, 10});
    UnitIndexer indexer(host_ref());

    EXPECT_EQ(1, indexer.num_universes());
    EXPECT_EQ(4, indexer.num_surfaces());
    EXPECT_EQ(10, indexer.num_volumes());

    EXPECT_EQ(SurfaceId(0),
              indexer.global_surface(UniverseId{0}, SurfaceId{0}));
    EXPECT_EQ(SurfaceId(3),
              indexer.global_surface(UniverseId{0}, SurfaceId{3}));

    EXPECT_EQ(VolumeId(0), indexer.global_volume(UniverseId{0}, VolumeId{0}));
    EXPECT_EQ(VolumeId(9), indexer.global_volume(UniverseId{0}, VolumeId{9}));

    auto local_s = indexer.local_surface(SurfaceId{0});
    EXPECT_EQ(UniverseId(0), local_s.universe);
    EXPECT_EQ(SurfaceId(0), local_s.surface);
    local_s = indexer.local_surface(SurfaceId{3});
    EXPECT_EQ(UniverseId(0), local_s.universe);
    EXPECT_EQ(SurfaceId(3), local_s.surface);

    auto local_v = indexer.local_volume(VolumeId{0});
    EXPECT_EQ(UniverseId(0), local_v.universe);
    EXPECT_EQ(VolumeId(0), local_v.volume);
    local_v = indexer.local_volume(VolumeId{3});
    EXPECT_EQ(UniverseId(0), local_v.universe);
    EXPECT_EQ(VolumeId(3), local_v.volume);
}

TEST_F(UnitIndexerTest, TEST_IF_CELERITAS_DEBUG(errors))
{
    set_data({0, 4}, {0, 10});
    UnitIndexer indexer(host_ref());

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

TEST_F(UnitIndexerTest, multi)
{
    // One universe has zero surfaces
    const std::vector<size_type> surfaces_per_uni{4, 1, 0, 1};
    const std::vector<size_type> cells_per_uni{1, 1, 1, 2};

    std::vector<size_type> surfaces = {0, 4, 5, 5, 6};
    std::vector<size_type> volumes  = {0, 1, 2, 3, 5};

    set_data(surfaces, volumes);
    UnitIndexer indexer(host_ref());

    EXPECT_EQ(6, indexer.num_surfaces());
    EXPECT_EQ(5, indexer.num_volumes());
    EXPECT_EQ(4, indexer.num_universes());

    unsigned int global_surface = 0;
    for (auto u : range(surfaces_per_uni.size()))

        for (auto s : range(surfaces_per_uni[u]))
        {
            auto local = indexer.local_surface(SurfaceId{global_surface});
            EXPECT_EQ(u, local.universe.unchecked_get());
            EXPECT_EQ(s, local.surface.unchecked_get());
            EXPECT_EQ(global_surface,
                      indexer.global_surface(local.universe, local.surface)
                          .unchecked_get());
            ++global_surface;
        }

    unsigned int global_volume = 0;
    for (auto u : range(cells_per_uni.size()))
    {
        for (auto c : range(cells_per_uni[u]))
        {
            auto local = indexer.local_volume(VolumeId{global_volume});
            EXPECT_EQ(u, local.universe.unchecked_get());
            EXPECT_EQ(c, local.volume.unchecked_get());
            EXPECT_EQ(global_volume,
                      indexer.global_volume(local.universe, local.volume)
                          .unchecked_get());
            ++global_volume;
        }
    }
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
