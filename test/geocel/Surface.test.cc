//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Surface.test.cc
//---------------------------------------------------------------------------//
#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/ScopedLogStorer.hh"
#include "corecel/cont/LabelIdMultiMapUtils.hh"
#include "corecel/io/Logger.hh"
#include "geocel/SurfaceParams.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/VolumeSurfaceView.hh"
#include "geocel/inp/Model.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using inp::Surface;
using inp::Surfaces;
using Boundary = Surface::Boundary;
using Interface = Surface::Interface;
using VolInstId = VolumeInstanceId;

//---------------------------------------------------------------------------//

//! Helper to create a boundary surface
Surface make_surface(std::string&& label, VolumeId vol)
{
    Surface surface;
    surface.label = std::move(label);
    surface.surface = vol;
    return surface;
}

//! Helper to create an interface surface
Surface make_surface(std::string&& label, VolInstId pre, VolInstId post)
{
    Surface surface;
    surface.label = std::move(label);
    surface.surface = Interface{pre, post};
    return surface;
}

//---------------------------------------------------------------------------//

class SurfacesTest : public ::celeritas::test::Test
{
  protected:
    /*!
     * Volumes: parent -> daughter [instance]
     * A -> B [0]
     * A -> C [1]
     * B -> C [2]
     * B -> C [3]
     * C -> D [4]
     * C -> E [5]
     */
    VolumeParams make_volume_params() const
    {
        return VolumeParams([] {
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
};

//! Construct for EM-only physics
TEST_F(SurfacesTest, none)
{
    SurfaceParams sp;
    EXPECT_TRUE(sp.empty());
    EXPECT_TRUE(sp.disabled());
    EXPECT_EQ(0, sp.num_surfaces());
    EXPECT_EQ(0, sp.labels().size());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(VolumeSurfaceView(sp.host_ref(), VolumeId{0}), DebugError);
    }
}

//! Construct for optical physics
TEST_F(SurfacesTest, none_but_volumes)
{
    auto volumes = this->make_volume_params();
    SurfaceParams sp{{}, volumes};
    EXPECT_TRUE(sp.empty());
    EXPECT_FALSE(sp.disabled());
    EXPECT_EQ(0, sp.num_surfaces());
    EXPECT_EQ(0, sp.labels().size());

    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{0});
        EXPECT_EQ(SurfaceId{}, vsv.boundary_id());
        EXPECT_FALSE(vsv.has_interface());
    }
}

TEST_F(SurfacesTest, errors)
{
    ScopedLogStorer scoped_log_{&celeritas::world_logger()};

    auto volumes = this->make_volume_params();
    // Duplicate boundary surface
    EXPECT_THROW(SurfaceParams(Surfaces{{
                                   make_surface("ok", VolumeId{1}),
                                   make_surface("bad", VolumeId{1}),
                               }},
                               volumes),
                 RuntimeError);

    // Duplicate interface surface
    EXPECT_THROW(
        SurfaceParams(Surfaces{{
                          make_surface("ok2", VolInstId{1}, VolInstId{2}),
                          make_surface("bad2", VolInstId{1}, VolInstId{2}),
                      }},
                      volumes),
        RuntimeError);

    static char const* const expected_log_messages[] = {
        "While processing surface 'bad'", "While processing surface 'bad2'"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"error", "error"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

TEST_F(SurfacesTest, borders)
{
    SurfaceParams sp{Surfaces{{
                         make_surface("b", VolumeId{1}),
                         make_surface("d", VolumeId{3}),
                         make_surface("e", VolumeId{4}),
                     }},
                     this->make_volume_params()};

    EXPECT_FALSE(sp.empty());
    EXPECT_FALSE(sp.disabled());
    EXPECT_EQ(3, sp.num_surfaces());
    static char const* const expected_labels[] = {"b", "d", "e"};
    EXPECT_VEC_EQ(expected_labels, get_multimap_labels(sp.labels()));

    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{0});
        EXPECT_EQ(VolumeId{0}, vsv.volume_id());
        EXPECT_EQ(SurfaceId{}, vsv.boundary_id());
        EXPECT_FALSE(vsv.has_interface());
        EXPECT_EQ(SurfaceId{}, vsv.find_interface(VolInstId{0}, VolInstId{0}));
    }
    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{1});
        EXPECT_EQ(SurfaceId{0}, vsv.boundary_id());
        EXPECT_FALSE(vsv.has_interface());
    }
    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{3});
        EXPECT_EQ(SurfaceId{1}, vsv.boundary_id());
        EXPECT_FALSE(vsv.has_interface());
    }
}

TEST_F(SurfacesTest, interfaces)
{
    SurfaceParams sp{Surfaces{{
                         make_surface("c2b", VolInstId{2}, VolInstId{0}),
                         make_surface("c2c2", VolInstId{2}, VolInstId{2}),
                         make_surface("b", VolumeId{1}),
                         make_surface("cc2", VolInstId{1}, VolInstId{2}),
                         make_surface("c3c", VolInstId{3}, VolInstId{1}),
                         make_surface("bc", VolInstId{0}, VolInstId{1}),
                         make_surface("bc2", VolInstId{0}, VolInstId{2}),
                         make_surface("ec", VolInstId{5}, VolInstId{1}),
                         make_surface("db", VolInstId{4}, VolInstId{1}),
                     }},
                     this->make_volume_params()};
    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{0});  // A -> any
        EXPECT_FALSE(vsv.has_interface());
    }
    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{1});  // B -> any
        EXPECT_EQ(SurfaceId{2}, vsv.boundary_id());
        EXPECT_TRUE(vsv.has_interface());
        EXPECT_EQ(SurfaceId{}, vsv.find_interface(VolInstId{0}, VolInstId{0}));
        EXPECT_EQ(SurfaceId{5}, vsv.find_interface(VolInstId{0}, VolInstId{1}));
        EXPECT_EQ(SurfaceId{6}, vsv.find_interface(VolInstId{0}, VolInstId{2}));
    }
    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{2});  // C -> any
        EXPECT_EQ(SurfaceId{0}, vsv.find_interface(VolInstId{2}, VolInstId{0}));
        EXPECT_EQ(SurfaceId{1}, vsv.find_interface(VolInstId{2}, VolInstId{2}));
        EXPECT_EQ(SurfaceId{3}, vsv.find_interface(VolInstId{1}, VolInstId{2}));
        EXPECT_EQ(SurfaceId{}, vsv.find_interface(VolInstId{2}, VolInstId{1}));
        EXPECT_EQ(SurfaceId{4}, vsv.find_interface(VolInstId{3}, VolInstId{1}));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
