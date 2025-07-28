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
 */
class VolumeTestBase : public ::celeritas::test::Test
{
  public:
    // Create a single volume A
    inp::Volumes make_single_volume_inp() const;

    // Create volumes A-E with instances 0 through 5.
    inp::Volumes make_complex_volume_inp() const;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
