//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceUtils.hh
//---------------------------------------------------------------------------//
#pragma once
#include <string>

#include "geocel/inp/Model.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

//! Helper to create a boundary surface
inline inp::Surface make_surface(std::string&& label, VolumeId vol)
{
    inp::Surface surface;
    surface.label = std::move(label);
    surface.surface = vol;
    return surface;
}

//! Helper to create an interface surface
inline inp::Surface
make_surface(std::string&& label, VolumeInstanceId pre, VolumeInstanceId post)
{
    inp::Surface surface;
    surface.label = std::move(label);
    surface.surface = inp::Surface::Interface{pre, post};
    return surface;
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
