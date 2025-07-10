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

#include "SurfaceTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class SurfacesTest : public SurfaceTestBase
{
  protected:
    using Surface = inp::Surface;
    using Surfaces = inp::Surfaces;
    using Boundary = Surface::Boundary;
    using Interface = Surface::Interface;
    using VolInstId = VolumeInstanceId;
};

TEST_F(SurfacesTest, none)
{
    SurfaceParams sp;
    EXPECT_TRUE(sp.empty());
    EXPECT_EQ(0, sp.num_surfaces());
    EXPECT_EQ(0, sp.labels().size());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(VolumeSurfaceView(sp.host_ref(), VolumeId{0}), DebugError);
    }
}

TEST_F(SurfacesTest, none_but_volumes)
{
    SurfaceParams sp{{}, volumes_};
    EXPECT_TRUE(sp.empty());
    EXPECT_EQ(0, sp.num_surfaces());
    EXPECT_EQ(0, sp.labels().size());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(VolumeSurfaceView(sp.host_ref(), VolumeId{0}), DebugError);
    }
}

TEST_F(SurfacesTest, errors)
{
    ScopedLogStorer scoped_log_{&celeritas::world_logger()};

    // Duplicate boundary surface
    EXPECT_THROW(SurfaceParams(Surfaces{{
                                   make_surface("ok", VolumeId{1}),
                                   make_surface("bad", VolumeId{1}),
                               }},
                               volumes_),
                 RuntimeError);

    // Duplicate interface surface
    EXPECT_THROW(
        SurfaceParams(Surfaces{{
                          make_surface("ok2", VolInstId{1}, VolInstId{2}),
                          make_surface("bad2", VolInstId{1}, VolInstId{2}),
                      }},
                      volumes_),
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
                     volumes_};

    EXPECT_FALSE(sp.empty());
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
    SurfaceParams sp{this->make_many_surfaces_inp(), volumes_};
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
