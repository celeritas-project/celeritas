//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "geocel/GenericGeoTestBase.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeGeoTraits.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
using OrangeTestBase = GenericGeoTestBase<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
