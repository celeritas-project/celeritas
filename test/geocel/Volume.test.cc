//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Volume.test.cc
//! Test VolumeParams and related utilities
//---------------------------------------------------------------------------//
#include "corecel/OpaqueIdUtils.hh"
#include "corecel/cont/LabelIdMultiMapUtils.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/VolumeToString.hh"
#include "geocel/VolumeVisitor.hh"
#include "geocel/inp/Model.hh"

#include "TestMacros.hh"
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

struct MaxVisitor
{
    VolumeParams::VolumeMap const& labels;
    std::unordered_map<VolumeId, int> max_depth;

    bool operator()(VolumeId id, int depth)
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
class VolumeTest : public ::celeritas::test::Test
{
  protected:
    VolumeParams volumes_;
};

//---------------------------------------------------------------------------//
class NoVolumeTest : public VolumeTest
{
};

/*!
 * No volumes, for unit testing
 */
TEST_F(NoVolumeTest, params)
{
    VolumeParams const& params = volumes_;
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
class SingleVolumeTest : public VolumeTest
{
  public:
    void SetUp()
    {
        volumes_ = VolumeParams([] {
            using namespace inp;
            Volumes in;
            in.volumes.push_back([] {
                Volume v;
                v.label = "A";
                v.material = GeoMatId{0};
                return v;
            }());
            in.world = VolumeId{0};
            return in;
        }());
    }
};

TEST_F(SingleVolumeTest, params)
{
    VolumeParams const& params = volumes_;

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
    VolumeToString to_string{volumes_};
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
    VolumeVisitor visit(volumes_);
    {
        NameVisitor nv{volumes_, {}};
        visit(nv, VolumeId{0});

        static std::string const expected_names[] = {"A"};
        EXPECT_VEC_EQ(expected_names, nv.names);
    }
}

//---------------------------------------------------------------------------//
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
class ComplexVolumeTest : public VolumeTest
{
  public:
    void SetUp()
    {
        volumes_ = VolumeParams([] {
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
            // Add 'null' (unused) instance
            in.volume_instances.push_back(VolumeInstance{});
            add_instance(VolumeId{4});  // 6 -> E

            // Top-level volume is zero
            in.world = VolumeId{0};

            return in;
        }());
    }
};

TEST_F(ComplexVolumeTest, params)
{
    VolumeParams const& params = volumes_;

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
    VolumeToString to_string{volumes_};
    EXPECT_EQ("A", to_string(VolumeId{0}));
    EXPECT_EQ("1", to_string(VolumeInstanceId{1}));
}

TEST_F(ComplexVolumeTest, visit)
{
    VolumeVisitor visit(volumes_);

    {
        NameVisitor nv{volumes_, {}};
        visit(nv, VolumeId{0});

        static std::string const expected_names[]
            = {"A", "B", "C", "D", "E", "C", "D", "E", "C", "D", "E"};
        EXPECT_VEC_EQ(expected_names, nv.names);
    }

    {
        NameVisitor nv{volumes_, {}};
        visit(nv, VolumeInstanceId{0});

        static std::string const expected_names[]
            = {"0:0", "1:2", "2:4", "2:6", "1:3", "2:4", "2:6"};
        EXPECT_VEC_EQ(expected_names, nv.names);
    }

    {
        MaxVisitor mpv{volumes_.volume_labels(), {}};
        visit(mpv, VolumeId{0});
        static std::string const expected_names[]
            = {"0:A", "1:B", "2:C", "3:D", "3:E"};
        EXPECT_VEC_EQ(expected_names, mpv.get_names());
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
