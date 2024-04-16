//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
using VecgeomTestBase = GenericGeoTestBase<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
