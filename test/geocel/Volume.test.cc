//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Volume.test.cc
//! Test VolumeParams and related utilities
//---------------------------------------------------------------------------//
#include <unordered_map>

#include "corecel/OpaqueIdUtils.hh"
#include "corecel/cont/LabelIdMultiMapUtils.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/VolumeToString.hh"
#include "geocel/VolumeVisitor.hh"

#include "TestMacros.hh"
#include "VolumeTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
struct NameVisitor
{
    VolumeParams const& vols;
    std::vector<std::string> names;

    bool operator()(VolumeId id, int)
    {
        names.push_back(vols.volume_labels().at(id).name);
        return true;
    }

    bool operator()(VolumeInstanceId id, int depth)
    {
        auto const& vlabels = vols.volume_instance_labels();
        std::ostringstream os;
        os << depth << ':' << vlabels.at(id).name;
        names.push_back(std::move(os).str());
        return true;
    }
};

template<class T = VolumeId>
struct MaxVisitor
{
    LabelIdMultiMap<T> const& labels;
    std::unordered_map<T, int> max_depth;

    bool operator()(T id, int depth)
    {
        auto&& [iter, inserted] = max_depth.insert({id, depth});
        if (!inserted)
        {
            if (iter->second >= depth)
            {
                // Already visited PV at this depth or more
                return false;
            }
            // Update the max depth
            iter->second = depth;
        }
        return true;
    }

    auto get_names() const
    {
        std::vector<std::string> names;
        for (auto&& [id, depth] : max_depth)
        {
            std::ostringstream os;
            os << depth << ':' << labels.at(id).name;
            names.push_back(std::move(os).str());
        }
        // Make reproducible across unordered map implementation
        std::sort(names.begin(), names.end());
        return names;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Note in the following tests:
 * - volumes are alphabetical (A, B, C...)
 * - volume instances are numeric (0, 1, 2...)
 */

//---------------------------------------------------------------------------//
class NoVolumeTest : public VolumeTestBase
{
  protected:
    std::shared_ptr<VolumeParams> build_volumes() const override
    {
        return std::make_shared<VolumeParams>();
    }
};

/*!
 * No volumes, for unit testing
 */
TEST_F(NoVolumeTest, params)
{
    VolumeParams const& params = this->volumes();
    EXPECT_TRUE(params.empty());
    EXPECT_EQ(0, params.num_volumes());
    EXPECT_EQ(VolumeId{}, params.world());
    EXPECT_EQ(0, params.depth());
}

TEST_F(NoVolumeTest, volume_to_string)
{
    VolumeToString to_string;
    EXPECT_EQ("<null>", to_string(VolumeId{}));
    EXPECT_EQ("<null>", to_string(VolumeInstanceId{}));
    EXPECT_EQ("v 1", to_string(VolumeId{1}));
    EXPECT_EQ("vi 2", to_string(VolumeInstanceId{2}));
}

//---------------------------------------------------------------------------//
/*!
 * Graph:
 *    A
 */
using SingleVolumeTest = SingleVolumeTestBase;

TEST_F(SingleVolumeTest, params)
{
    VolumeParams const& params = this->volumes();

    EXPECT_FALSE(params.empty());
    EXPECT_EQ(1, params.num_volumes());
    EXPECT_EQ(0, params.num_volume_instances());
    EXPECT_EQ(VolumeId{0}, params.world());
    EXPECT_EQ(0, params.depth());
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

TEST_F(SingleVolumeTest, volume_to_string)
{
    VolumeToString to_string{this->volumes()};
    EXPECT_EQ("<null>", to_string(VolumeId{}));
    EXPECT_EQ("<null>", to_string(VolumeInstanceId{}));
    EXPECT_EQ("A", to_string(VolumeId{0}));

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(to_string(VolumeId{1}), DebugError);
    }
}

TEST_F(SingleVolumeTest, visit)
{
    VolumeVisitor visit(this->volumes());
    {
        NameVisitor nv{this->volumes(), {}};
        visit(nv, VolumeId{0});

        static std::string const expected_names[] = {"A"};
        EXPECT_VEC_EQ(expected_names, nv.names);
    }
}

//---------------------------------------------------------------------------//
using ComplexVolumeTest = ComplexVolumeTestBase;

TEST_F(ComplexVolumeTest, params)
{
    VolumeParams const& params = this->volumes();

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

TEST_F(ComplexVolumeTest, volume_to_string)
{
    VolumeToString to_string{this->volumes()};
    EXPECT_EQ("A", to_string(VolumeId{0}));
    EXPECT_EQ("1", to_string(VolumeInstanceId{1}));
}

TEST_F(ComplexVolumeTest, visit)
{
    VolumeVisitor visit(this->volumes());

    {
        NameVisitor nv{this->volumes(), {}};
        visit(nv, VolumeId{0});

        static std::string const expected_names[]
            = {"A", "B", "C", "D", "E", "C", "D", "E", "C", "D", "E"};
        EXPECT_VEC_EQ(expected_names, nv.names);
    }

    {
        NameVisitor nv{this->volumes(), {}};
        visit(nv, VolumeInstanceId{0});

        static std::string const expected_names[]
            = {"0:0", "1:2", "2:4", "2:6", "1:3", "2:4", "2:6"};
        EXPECT_VEC_EQ(expected_names, nv.names);
    }

    {
        MaxVisitor<> mpv{this->volumes().volume_labels(), {}};
        visit(mpv, VolumeId{0});
        static std::string const expected_names[]
            = {"0:A", "1:B", "2:C", "3:D", "3:E"};
        EXPECT_VEC_EQ(expected_names, mpv.get_names());
    }
}

//---------------------------------------------------------------------------//
using MultiLevelTest = MultiLevelVolumeTestBase;

TEST_F(MultiLevelTest, visit)
{
    auto const& vols = this->volumes();
    auto const world_vi
        = vols.volume_instance_labels().find_unique("world_PV");
    VolumeVisitor visit(vols);

    {
        NameVisitor nv{vols, {}};
        visit(nv, vols.world());

        static std::string const expected_names[] = {
            "world",
            "box",
            "sph",
            "sph",
            "tri",
            "sph",
            "box",
            "sph",
            "sph",
            "tri",
            "box",
            "sph",
            "sph",
            "tri",
            "box_refl",
            "sph_refl",
            "sph_refl",
            "tri_refl",
        };
        EXPECT_VEC_EQ(expected_names, nv.names);
    }

    {
        NameVisitor nv{vols, {}};
        visit(nv, world_vi);

        static std::string const expected_names[] = {
            "0:world_PV",
            "1:topbox1",
            "2:boxsph1",
            "2:boxsph2",
            "2:boxtri",
            "1:topsph1",
            "1:topbox2",
            "2:boxsph1",
            "2:boxsph2",
            "2:boxtri",
            "1:topbox3",
            "2:boxsph1",
            "2:boxsph2",
            "2:boxtri",
            "1:topbox4",
            "2:boxsph1",
            "2:boxsph2",
            "2:boxtri",
        };
        EXPECT_VEC_EQ(expected_names, nv.names);
    }

    {
        MaxVisitor<> mpv{vols.volume_labels(), {}};
        visit(mpv, vols.world());
        static std::string const expected_names[] = {
            "0:world",
            "1:box",
            "1:box_refl",
            "2:sph",
            "2:sph_refl",
            "2:tri",
            "2:tri_refl",
        };
        EXPECT_VEC_EQ(expected_names, mpv.get_names());
    }

    {
        MaxVisitor<VolumeInstanceId> mpv{vols.volume_instance_labels(), {}};
        visit(mpv, world_vi);
        static std::string const expected_names[] = {
            "0:world_PV",
            "1:topbox1",
            "1:topbox2",
            "1:topbox3",
            "1:topbox4",
            "1:topsph1",
            "2:boxsph1",
            "2:boxsph1",
            "2:boxsph2",
            "2:boxsph2",
            "2:boxtri",
            "2:boxtri",
        };
        EXPECT_VEC_EQ(expected_names, mpv.get_names());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
