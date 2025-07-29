//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceTestBase.cc
//---------------------------------------------------------------------------//
#include "SurfaceTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using VolInstId = VolumeInstanceId;

//---------------------------------------------------------------------------//
/*!
 * Construct volumes.
 */
void SurfaceTestBase::SetUp()
{
    volumes_ = VolumeParams([] {
        using namespace inp;
        Volumes in;

        // Helper to create volumes
        auto add_volume
            = [&in](std::string label, std::vector<VolInstId> children) {
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

        add_volume("A", {VolInstId{0}, VolInstId{1}});
        add_volume("B", {VolInstId{2}, VolInstId{3}});
        add_volume("C", {VolInstId{4}, VolInstId{5}});
        add_volume("D", {});
        add_volume("E", {});

        add_instance(VolumeId{1});  // 0 -> B
        add_instance(VolumeId{2});  // 1 -> C
        add_instance(VolumeId{2});  // 2 -> C
        add_instance(VolumeId{2});  // 3 -> C
        add_instance(VolumeId{3});  // 4 -> D
        add_instance(VolumeId{4});  // 5 -> E

        return in;
    }());
}

//---------------------------------------------------------------------------//
/*!
 * Create many-connected surfaces input.
 */
inp::Surfaces SurfaceTestBase::make_many_surfaces_inp() const
{
    CELER_VALIDATE(
        volumes_.num_volumes() == 5 && volumes_.num_volume_instances() == 6,
        << "volumes were not constructed in SetUp");

    return inp::Surfaces{{
        make_surface("c2b", VolInstId{2}, VolInstId{0}),
        make_surface("c2c2", VolInstId{2}, VolInstId{2}),
        make_surface("b", VolumeId{1}),
        make_surface("cc2", VolInstId{1}, VolInstId{2}),
        make_surface("c3c", VolInstId{3}, VolInstId{1}),
        make_surface("bc", VolInstId{0}, VolInstId{1}),
        make_surface("bc2", VolInstId{0}, VolInstId{2}),
        make_surface("ec", VolInstId{5}, VolInstId{1}),
        make_surface("db", VolInstId{4}, VolInstId{1}),
    }};
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
