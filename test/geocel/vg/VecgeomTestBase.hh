//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/GenericGeoTestBase.hh"
#include "geocel/vg/VecgeomData.hh"
#include "geocel/vg/VecgeomGeoTraits.hh"
#include "geocel/vg/VecgeomParams.hh"
#include "geocel/vg/VecgeomTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class VecgeomTestBase : public GenericGeoTestBase<VecgeomParams>
{
  public:
    // TODO: surface normals do NOT currently work
    bool supports_surface_normal() const override { return false; }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
