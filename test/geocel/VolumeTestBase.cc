//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeTestBase.cc
//---------------------------------------------------------------------------//
#include "VolumeTestBase.hh"

#include "corecel/Assert.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using VolInstId = VolumeInstanceId;

void VolumeTestBase::SetUp()
{
    volumes_ = this->build_volumes();
    CELER_ASSERT(volumes_);
}

//---------------------------------------------------------------------------//

VolumeParams const& VolumeTestBase::volumes() const
{
    CELER_EXPECT(volumes_);
    return *volumes_;
}

//---------------------------------------------------------------------------//

std::shared_ptr<VolumeParams> SingleVolumeTestBase::build_volumes() const
{
    using namespace inp;
    Volumes in;
    in.volumes.push_back([] {
        Volume v;
        v.label = "A";
        v.material = GeoMatId{0};
        return v;
    }());
    in.world = VolumeId{0};
    return std::make_shared<VolumeParams>(std::move(in));
}

//---------------------------------------------------------------------------//

std::shared_ptr<VolumeParams> ComplexVolumeTestBase::build_volumes() const
{
    using namespace inp;
    Volumes in;

    // Helper to create volumes
    auto add_volume = [&in](std::string label, std::vector<VolInstId> children) {
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
    add_volume("A", {VolInstId{0}, VolInstId{1}});
    add_volume("B", {VolInstId{2}, VolInstId{3}});
    add_volume("C", {VolInstId{4}, VolInstId{6}});
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

    return std::make_shared<VolumeParams>(std::move(in));
}

//---------------------------------------------------------------------------//

std::shared_ptr<VolumeParams> OpticalVolumeTestBase::build_volumes() const
{
    using namespace inp;
    Volumes in;
    in.volumes = {
        {"lar_sphere", GeoMatId{1}, {}},
        {"tube1_mid", GeoMatId{2}, {}},
        {"tube2", GeoMatId{2}, {}},
        {
            "world",
            GeoMatId{3},
            {
                VolumeInstanceId{0},
                VolumeInstanceId{1},
                VolumeInstanceId{2},
                VolumeInstanceId{3},
            },
        },
    };

    // 2) Physical‐volume instances
    in.volume_instances = {
        {"lar_pv", VolumeId{0}},
        {"tube2_below_pv", VolumeId{2}},
        {"tube1_mid_pv", VolumeId{1}},
        {"tube2_above_pv", VolumeId{2}},
        {"world_PV", VolumeId{3}},
    };

    in.world = VolumeId{3};
    return std::make_shared<VolumeParams>(std::move(in));
}

//---------------------------------------------------------------------------//

std::shared_ptr<VolumeParams> MultiLevelVolumeTestBase::build_volumes() const
{
    using namespace inp;
    Volumes in;
    in.volumes = {
        {"sph", GeoMatId{0}, {}},
        {"tri", GeoMatId{0}, {}},
        {"box",
         GeoMatId{1},
         {VolumeInstanceId{0}, VolumeInstanceId{1}, VolumeInstanceId{2}}},
        {"world",
         GeoMatId{0},
         {VolumeInstanceId{3},
          VolumeInstanceId{4},
          VolumeInstanceId{5},
          VolumeInstanceId{6},
          VolumeInstanceId{10}}},
        {"box_refl",
         GeoMatId{1},
         {VolumeInstanceId{7}, VolumeInstanceId{8}, VolumeInstanceId{9}}},
        {"sph_refl", GeoMatId{0}, {}},
        {"tri_refl", GeoMatId{0}, {}},
    };

    in.volume_instances = {
        {Label{"boxsph1", "0"}, VolumeId{0}},
        {Label{"boxsph2", "0"}, VolumeId{0}},
        {Label{"boxtri", "0"}, VolumeId{1}},
        {"topbox1", VolumeId{2}},
        {"topsph1", VolumeId{0}},
        {"topbox2", VolumeId{2}},
        {"topbox3", VolumeId{2}},
        {Label{"boxsph1", "1"}, VolumeId{5}},
        {Label{"boxsph2", "1"}, VolumeId{5}},
        {Label{"boxtri", "1"}, VolumeId{6}},
        {"topbox4", VolumeId{4}},
        {"world_PV", VolumeId{3}},
    };

    in.world = VolumeId{3};
    return std::make_shared<VolumeParams>(std::move(in));
}

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
