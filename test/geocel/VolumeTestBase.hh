//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/inp/Model.hh"

#include "Test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Create inputs for different geometry hierarchies.
 *
 * The "single volume" input constructs a single volume A.
 *
 * The "complex volume" input constructs volumes A through E with three
 * instances of C (one inside A, two inside B), placing them in the hierarchy
 * with the following volume instances:
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
 * The "optical surfaces" test creates surfaces from `optical-surfaces.gdml`.
 * \verbatim
 * world      -> lar_sphere   "lar_pv"
   world      -> tube2        "tube2_below_pv"
   world      -> tube1_mid    "tube1_mid_pv"
   world      -> tube2        "tube2_above_pv"
 * \endverbatim
 *
 * The multi-level representation includes reflection and is:
 * \verbatim
   box       -> sph        "boxsph1:0"
   box       -> sph        "boxsph2:0"
   box       -> tri        "boxtri:0"
   world     -> box        "topbox1"
   world     -> sph        "topsph1"
   world     -> box        "topbox2"
   world     -> box        "topbox3"
   world     -> box_refl   "topbox4"
   box_refl  -> sph_refl   "boxsph1:1"
   box_refl  -> sph_refl   "boxsph2:1"
   box_refl  -> tri_refl   "boxtri:1"
 * \endverbatim
 */
class VolumeTestBase : public ::celeritas::test::Test
{
  public:
    // Create a single volume A
    inp::Volumes make_single_volume_inp() const;

    // Create volumes A-E with instances 0 through 5.
    inp::Volumes make_complex_volume_inp() const;

    // Create surfaces from the optical surfaces GDML
    inp::Volumes make_optical_volume_inp() const;

    // Create multi-level volume output
    inp::Volumes make_multi_level_volume_inp() const;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
