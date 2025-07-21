//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Volume.test.cc
//---------------------------------------------------------------------------//
#include "corecel/OpaqueIdUtils.hh"
#include "corecel/cont/LabelIdMultiMapUtils.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Note in the following tests:
 * - volumes are alphabetical (A, B, C...)
 * - volume instances are numeric (0, 1, 2...)
 */
using VolumeTest = ::celeritas::test::Test;

/*!
 * No volumes, for unit testing
 */
TEST_F(VolumeTest, no_volumes)
{
    VolumeParams const params;
    EXPECT_TRUE(params.empty());
    EXPECT_EQ(0, params.num_volumes());
}

/*!
 * Graph:
 *    A
 */
TEST_F(VolumeTest, single_volume)
{
    VolumeParams const params([] {
        using namespace inp;
        Volumes in;
        Volume v;
        v.label = "A";
        v.material = GeoMatId{0};
        in.volumes.push_back(v);
        return in;
    }());

    EXPECT_FALSE(params.empty());
    EXPECT_EQ(1, params.num_volumes());
    EXPECT_EQ(0, params.num_volume_instances());
    EXPECT_EQ(1, params.volume_labels().size());
    EXPECT_EQ(0, params.volume_instance_labels().size());

    // Check that volume 0 is correctly mapped
    VolumeId vol_id{0};
    EXPECT_TRUE(params.volume_labels().find_unique("A") == vol_id);

    // Verify material assignment
    EXPECT_EQ(GeoMatId{0}, params.material(vol_id));

    // A single volume should have no parents or children
    EXPECT_TRUE(params.parents(vol_id).empty());
    EXPECT_TRUE(params.children(vol_id).empty());

    // Test out-of-bounds access should assert
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(params.material(VolumeId{1}), DebugError);
        EXPECT_THROW(params.parents(VolumeId{1}), DebugError);
        EXPECT_THROW(params.children(VolumeId{1}), DebugError);
        EXPECT_THROW(params.volume(VolumeInstanceId{0}), DebugError);
    }
}

/*!
 * Graph:
 * A -> B [0]
 * A -> C [1]
 * B -> C [2]
 * B -> C [3]
 * C -> D [4]
 * C -> E [6]
 *
 * Where bracketed values are volume instance IDs.
 */
TEST_F(VolumeTest, complex_hierarchy)
{
    VolumeParams const params([] {
        using namespace inp;
        Volumes in;

        // Helper to create volumes
        auto add_volume = [&in](std::string label,
                                std::vector<VolumeInstanceId> children) {
            Volume v;
            v.label = std::move(label);
            v.material = id_cast<GeoMatId>(in.volumes.size());
            v.children = std::move(children);
            in.volumes.push_back(v);
        };
        auto add_instance = [&in](VolumeId vol_id) {
            VolumeInstance vi;
            vi.label = std::to_string(in.volume_instances.size());
            vi.volume = vol_id;
            in.volume_instances.push_back(vi);
        };

        // Create volumes A, B, C, D, E
        add_volume("A", {VolumeInstanceId{0}, VolumeInstanceId{1}});
        add_volume("B", {VolumeInstanceId{2}, VolumeInstanceId{3}});
        add_volume("C", {VolumeInstanceId{4}, VolumeInstanceId{6}});
        add_volume("D", {});
        add_volume("E", {});

        // Create volume instances
        add_instance(VolumeId{1});  // 0 -> B
        add_instance(VolumeId{2});  // 1 -> C
        add_instance(VolumeId{2});  // 2 -> C
        add_instance(VolumeId{2});  // 3 -> C
        add_instance(VolumeId{3});  // 4 -> D
        // Add 'null' instance
        in.volume_instances.push_back(VolumeInstance{});
        add_instance(VolumeId{4});  // 6 -> E

        return in;
    }());

    static std::string const expected_volume_labels[]
        = {"A", "B", "C", "D", "E"};
    static std::string const expected_volume_instance_labels[]
        = {"0", "1", "2", "3", "4", "", "6"};

    // Check volume labels
    EXPECT_VEC_EQ(expected_volume_labels,
                  get_multimap_labels(params.volume_labels()));
    EXPECT_VEC_EQ(expected_volume_instance_labels,
                  get_multimap_labels(params.volume_instance_labels()));

    std::vector<std::vector<int>> children;
    std::vector<std::vector<int>> parents;
    std::vector<int> geo_mat;

    // Loop over all volumes to collect children and parents
    for (auto vol_id : range(VolumeId(params.num_volumes())))
    {
        children.push_back(id_to_int(params.children(vol_id)));
        parents.push_back(id_to_int(params.parents(vol_id)));
        geo_mat.push_back(id_to_int(params.material(vol_id)));
    }

    static std::vector<int> const expected_children[]
        = {{0, 1}, {2, 3}, {4, 6}, {}, {}};
    static std::vector<int> const expected_parents[]
        = {{}, {0}, {1, 2, 3}, {4}, {6}};
    static int const expected_geo_mat[] = {0, 1, 2, 3, 4};
    EXPECT_VEC_EQ(expected_children, children);
    EXPECT_VEC_EQ(expected_parents, parents);
    EXPECT_VEC_EQ(expected_geo_mat, geo_mat);

    // Check volume instance to volume mappings
    std::vector<int> volume_mapping;
    for (auto inst_id : range(VolumeInstanceId(params.num_volume_instances())))
    {
        volume_mapping.push_back(id_to_int(params.volume(inst_id)));
    }

    static int const expected_volume_mapping[] = {1, 2, 2, 2, 3, -1, 4};
    EXPECT_VEC_EQ(expected_volume_mapping, volume_mapping);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
