//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/SurfaceTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"

#include "Test.hh"
#include "VolumeTestBase.hh"

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
/*!
 * Construct volume params and emit surface input on request.
 *
 * The "many surface" constructor builds the following surfaces:
 * \verbatim
    c2b : interface 2 -> 0
    c2c2: interface 2 -> 2
    b   : boundary for A
    cc2 : interface 1 -> 2
    c3c : interface 3 -> 1
    bc  : interface 0 -> 1
    bc2 : interface 0 -> 2
    ec  : interface 5 -> 1
    db  : interface 4 -> 1
 * \endverbatim
 *
 * The optical surfaces are:
 * \verbatim
    sphere_skin   : boundary for 0 (lar_sphere)
    tube2_skin    : boundary for 1 (tube2)
    below_to_1    : interface 1 -> 2 (tube2_below_pv -> tube1_mid_pv)
    mid_to_below  : interface 2 -> 1 (tube1_mid_pv -> tube2_below_pv)
    mid_to_above  : interface 2 -> 3 (tube1_mid_pv -> tube2_above_pv)
 * \endverbatim
 */
class SurfaceTestBase : public VolumeTestBase
{
  public:
    // Create many-connected surfaces input and corresponding volumes
    inp::Surfaces make_many_surfaces_inp();

    // Create surfaces from `optical-surfaces.gdml`
    inp::Surfaces make_optical_surfaces_inp();

    //! Access volumes created by surfaces input
    VolumeParams const& volumes() const { return volumes_; }

  protected:
    VolumeParams volumes_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
