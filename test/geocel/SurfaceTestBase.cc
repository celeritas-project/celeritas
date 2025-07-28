//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceTestBase.cc
//---------------------------------------------------------------------------//
#include "SurfaceTestBase.hh"

#include "corecel/Assert.hh"
#include "geocel/SurfaceParams.hh"
#include "geocel/inp/Model.hh"

#include "SurfaceUtils.hh"

namespace celeritas
{
namespace test
{
using VolInstId = VolumeInstanceId;
//---------------------------------------------------------------------------//

void SurfaceTestBase::SetUp()
{
    // Call parent setup to initialize volumes_
    VolumeTestBase::SetUp();

    // Now build the surfaces
    surfaces_ = this->build_surfaces();
    CELER_ASSERT(surfaces_);
}

//---------------------------------------------------------------------------//

SurfaceParams const& SurfaceTestBase::surfaces() const
{
    CELER_EXPECT(surfaces_);
    return *surfaces_;
}

//---------------------------------------------------------------------------//

std::shared_ptr<SurfaceParams> ManySurfacesTestBase::build_surfaces() const
{
    inp::Surfaces in{{
        make_surface("c2b", VolInstId{2}, VolInstId{0}),
        make_surface("c2c2", VolInstId{2}, VolInstId{2}),
        make_surface("b", VolumeId{1}),
        make_surface("cc2", VolInstId{1}, VolInstId{2}),
        make_surface("c3c", VolInstId{3}, VolInstId{1}),
        make_surface("bc", VolInstId{0}, VolInstId{1}),
        make_surface("bc2", VolInstId{0}, VolInstId{2}),
        make_surface("ec", VolInstId{6}, VolInstId{1}),
        make_surface("db", VolInstId{4}, VolInstId{1}),
    }};

    return std::make_shared<SurfaceParams>(std::move(in), this->volumes());
}

//---------------------------------------------------------------------------//
std::shared_ptr<SurfaceParams> OpticalSurfacesTestBase::build_surfaces() const
{
    inp::Surfaces in{{
        make_surface("sphere_skin", VolumeId{0}),
        make_surface("tube2_skin", VolumeId{2}),
        make_surface("below_to_1", VolumeInstanceId{1}, VolumeInstanceId{2}),
        make_surface("mid_to_below", VolumeInstanceId{2}, VolumeInstanceId{1}),
        make_surface("mid_to_above", VolumeInstanceId{2}, VolumeInstanceId{3}),
    }};

    return std::make_shared<SurfaceParams>(std::move(in), this->volumes());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
