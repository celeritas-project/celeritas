//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/UniverseIndexer.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/UniverseIndexer.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "geocel/Types.hh"

#include "celeritas_test.hh"

using UniverseIndexer = celeritas::detail::UniverseIndexer;

namespace celeritas
{
namespace test
{
class UniverseIndexerTest : public Test
{
  public:
    using CollectionHostRef
        = UniverseIndexerData<Ownership::const_reference, MemSpace::host>;
    using VecSize = std::vector<size_type>;

    CollectionHostRef const& host_ref() const { return data_.host_ref(); }

    void set_data(VecSize surfaces, VecSize volumes)
    {
        UniverseIndexerData<Ownership::value, MemSpace::host> data;

        auto cb_s = make_builder(&data.surfaces);
        cb_s.insert_back(surfaces.begin(), surfaces.end());

        auto cb_v = make_builder(&data.volumes);
        cb_v.insert_back(volumes.begin(), volumes.end());

        data_ = CollectionMirror<UniverseIndexerData>{std::move(data)};
    }

  protected:
    CollectionMirror<UniverseIndexerData> data_;
};

//---------------------------------------------------------------------------//

TEST_F(UniverseIndexerTest, single)
{
    this->set_data({0, 4}, {0, 10});
    UniverseIndexer indexer(this->host_ref());

    EXPECT_EQ(1, indexer.num_universes());
    EXPECT_EQ(4, indexer.num_surfaces());
    EXPECT_EQ(10, indexer.num_volumes());

    EXPECT_EQ(ImplSurfaceId(0),
              indexer.global_surface(UniverseId{0}, LocalSurfaceId{0}));
    EXPECT_EQ(ImplSurfaceId(3),
              indexer.global_surface(UniverseId{0}, LocalSurfaceId{3}));

    EXPECT_EQ(ImplVolumeId(0),
              indexer.global_volume(UniverseId{0}, LocalVolumeId{0}));
    EXPECT_EQ(ImplVolumeId(9),
              indexer.global_volume(UniverseId{0}, LocalVolumeId{9}));

    auto local_s = indexer.local_surface(ImplSurfaceId{0});
    EXPECT_EQ(UniverseId(0), local_s.universe);
    EXPECT_EQ(LocalSurfaceId(0), local_s.surface);
    local_s = indexer.local_surface(ImplSurfaceId{3});
    EXPECT_EQ(UniverseId(0), local_s.universe);
    EXPECT_EQ(LocalSurfaceId(3), local_s.surface);

    auto local_v = indexer.local_volume(ImplVolumeId{0});
    EXPECT_EQ(UniverseId(0), local_v.universe);
    EXPECT_EQ(LocalVolumeId(0), local_v.volume);
    local_v = indexer.local_volume(ImplVolumeId{3});
    EXPECT_EQ(UniverseId(0), local_v.universe);
    EXPECT_EQ(LocalVolumeId(3), local_v.volume);
}

TEST_F(UniverseIndexerTest, TEST_IF_CELERITAS_DEBUG(errors))
{
    this->set_data({0, 4}, {0, 10});
    UniverseIndexer indexer(this->host_ref());

    EXPECT_THROW(indexer.global_surface(UniverseId{0}, LocalSurfaceId{4}),
                 DebugError);
    EXPECT_THROW(indexer.global_surface(UniverseId{1}, LocalSurfaceId{0}),
                 DebugError);
    EXPECT_THROW(indexer.global_volume(UniverseId{0}, LocalVolumeId{10}),
                 DebugError);
    EXPECT_THROW(indexer.global_volume(UniverseId{1}, LocalVolumeId{0}),
                 DebugError);

    EXPECT_THROW(indexer.local_surface(ImplSurfaceId(4)), DebugError);
    EXPECT_THROW(indexer.local_volume(ImplVolumeId(10)), DebugError);
}

TEST_F(UniverseIndexerTest, multi)
{
    // One universe has zero surfaces
    std::vector<size_type> const surfaces_per_uni{4, 1, 0, 1};
    std::vector<size_type> const cells_per_uni{1, 1, 1, 2};

    std::vector<size_type> surfaces = {0, 4, 5, 5, 6};
    std::vector<size_type> volumes = {0, 1, 2, 3, 5};

    this->set_data(surfaces, volumes);
    UniverseIndexer indexer(this->host_ref());

    EXPECT_EQ(6, indexer.num_surfaces());
    EXPECT_EQ(5, indexer.num_volumes());
    EXPECT_EQ(4, indexer.num_universes());

    unsigned int global_surface = 0;
    for (auto u : range(surfaces_per_uni.size()))

        for (auto s : range(surfaces_per_uni[u]))
        {
            auto local = indexer.local_surface(ImplSurfaceId{global_surface});
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
            auto local = indexer.local_volume(ImplVolumeId{global_volume});
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
}  // namespace test
}  // namespace celeritas
