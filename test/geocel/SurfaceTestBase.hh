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
 * Construct volume params at setup time.
 *
 * It constructs volumes A through E with three instances of C (one inside A,
 * two inside B), placing them in the hierarchy with the following volume
 * instances:
 * \verbatim
   {parent} -> {daughter} "{volume instance label}"
     A -> B "0"
     A -> C "1"
     B -> C "2"
     B -> C "3"
     C -> D "4"
     C -> E "5"
 * \endverbatim
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
 */
class SurfaceTestBase : public ::celeritas::test::Test
{
  public:
    // Construct volumes
    void SetUp() override;

    // Create many-connected surfaces input
    inp::Surfaces make_many_surfaces_inp() const;

  protected:
    VolumeParams volumes_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
