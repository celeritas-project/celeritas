//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceTestBase.cc
//---------------------------------------------------------------------------//
#include "SurfaceTestBase.hh"

#include "geocel/VolumeParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using VolInstId = VolumeInstanceId;

//---------------------------------------------------------------------------//
/*!
 * Create many-connected surfaces input.
 */
inp::Surfaces SurfaceTestBase::make_many_surfaces_inp()
{
    volumes_ = VolumeParams(this->make_complex_volume_inp());

    return inp::Surfaces{{
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
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
