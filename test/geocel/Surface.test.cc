//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Surface.test.cc
//---------------------------------------------------------------------------//
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
#include "SurfaceUtils.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using VolInstId = VolumeInstanceId;

// Test class for checking surface construction errors
class SurfaceErrorsTest : public ComplexVolumeTestBase
{
    // This class only needs volumes for testing errors
    // It doesn't need to build any surfaces
};

TEST_F(SurfaceErrorsTest, errors)
{
    ScopedLogStorer scoped_log_{&celeritas::world_logger()};

    // Duplicate boundary surface
    EXPECT_THROW(SurfaceParams(inp::Surfaces{{
                                   make_surface("ok", VolumeId{1}),
                                   make_surface("bad", VolumeId{1}),
                               }},
                               this->volumes()),
                 RuntimeError);

    // Duplicate interface surface
    EXPECT_THROW(
        SurfaceParams(inp::Surfaces{{
                          make_surface("ok2", VolInstId{1}, VolInstId{2}),
                          make_surface("bad2", VolInstId{1}, VolInstId{2}),
                      }},
                      this->volumes()),
        RuntimeError);

    static char const* const expected_log_messages[] = {
        "While processing surface 'bad'", "While processing surface 'bad2'"};
    EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
    static char const* const expected_log_levels[] = {"error", "error"};
    EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
}

//---------------------------------------------------------------------------//
//! Construct for EM-only physics
class NoSurfacesTest : public virtual VolumeTestBase, public SurfaceTestBase
{
  protected:
    std::shared_ptr<VolumeParams> build_volumes() const override
    {
        return std::make_shared<VolumeParams>();
    }

    std::shared_ptr<SurfaceParams> build_surfaces() const override
    {
        CELER_ENSURE(this->VolumeTestBase::volumes().empty());
        return std::make_shared<SurfaceParams>();
    }
};

TEST_F(NoSurfacesTest, params)
{
    auto const& sp = this->surfaces();
    EXPECT_TRUE(sp.empty());
    EXPECT_TRUE(sp.disabled());
    EXPECT_EQ(0, sp.num_surfaces());
    EXPECT_EQ(0, sp.labels().size());
}

TEST_F(NoSurfacesTest, TEST_IF_CELERITAS_DEBUG(vs_view))
{
    auto const& sp = this->surfaces();
    EXPECT_THROW(VolumeSurfaceView(sp.host_ref(), VolumeId{0}), DebugError);
}

//---------------------------------------------------------------------------//
//! Construct for optical physics
class NoSurfacesWithVolsTest : public SingleVolumeTestBase,
                               public SurfaceTestBase
{
  protected:
    std::shared_ptr<SurfaceParams> build_surfaces() const override
    {
        return std::make_shared<SurfaceParams>(
            inp::Surfaces{}, this->SingleVolumeTestBase::volumes());
    }
};

TEST_F(NoSurfacesWithVolsTest, params)
{
    auto const& sp = this->surfaces();
    EXPECT_TRUE(sp.empty());
    EXPECT_FALSE(sp.disabled());
    EXPECT_EQ(0, sp.num_surfaces());
    EXPECT_EQ(0, sp.labels().size());
}

TEST_F(NoSurfacesWithVolsTest, vs_view)
{
    VolumeSurfaceView vsv(this->surfaces().host_ref(), VolumeId{0});
    EXPECT_EQ(SurfaceId{}, vsv.boundary_id());
    EXPECT_FALSE(vsv.has_interface());
}

//---------------------------------------------------------------------------//
//! Construct with just borders
class BorderSurfacesTest : public ComplexVolumeTestBase, public SurfaceTestBase
{
  protected:
    std::shared_ptr<SurfaceParams> build_surfaces() const override
    {
        inp::Surfaces in{{
            make_surface("b", VolumeId{1}),
            make_surface("d", VolumeId{3}),
            make_surface("e", VolumeId{4}),
        }};
        return std::make_shared<SurfaceParams>(std::move(in), this->volumes());
    }
};

TEST_F(BorderSurfacesTest, params)
{
    auto const& sp = this->surfaces();
    EXPECT_FALSE(sp.empty());
    EXPECT_FALSE(sp.disabled());
    EXPECT_EQ(3, sp.num_surfaces());
    static char const* const expected_labels[] = {"b", "d", "e"};
    EXPECT_VEC_EQ(expected_labels, get_multimap_labels(sp.labels()));
}

TEST_F(BorderSurfacesTest, vs_view)
{
    auto const& sp = this->surfaces();
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

//---------------------------------------------------------------------------//
//! Construct with interfaces too
using ManySurfacesTest = ManySurfacesTestBase;

TEST_F(ManySurfacesTest, vs_view)
{
    auto const& sp = this->surfaces();
    // We know there are 5 volumes in the complex volume test
    ASSERT_EQ(5, this->ComplexVolumeTestBase::volumes().num_volumes());
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
//! Construct with optical interfaces
using OpticalSurfacesTest = OpticalSurfacesTestBase;

TEST_F(OpticalSurfacesTest, params)
{
    auto const& sp = this->surfaces();
    EXPECT_FALSE(sp.empty());
    EXPECT_FALSE(sp.disabled());
    EXPECT_EQ(5, sp.num_surfaces());
    static char const* const expected_labels[] = {
        "sphere_skin",
        "tube2_skin",
        "below_to_1",
        "mid_to_below",
        "mid_to_above",
    };
    EXPECT_VEC_EQ(expected_labels, get_multimap_labels(sp.labels()));
}

TEST_F(OpticalSurfacesTest, vs_view)
{
    auto const& sp = this->surfaces();
    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{0});  // lar_pv
        EXPECT_EQ(SurfaceId{0}, vsv.boundary_id());
        EXPECT_FALSE(vsv.has_interface());
    }
    {
        VolumeSurfaceView vsv(sp.host_ref(), VolumeId{1});  // tube1_mid
        EXPECT_EQ(SurfaceId{}, vsv.boundary_id());
        EXPECT_TRUE(vsv.has_interface());
        EXPECT_EQ(SurfaceId{3}, vsv.find_interface(VolInstId{2}, VolInstId{1}));
        EXPECT_EQ(SurfaceId{4}, vsv.find_interface(VolInstId{2}, VolInstId{3}));
        EXPECT_EQ(SurfaceId{}, vsv.find_interface(VolInstId{2}, VolInstId{2}));
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
