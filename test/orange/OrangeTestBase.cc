//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/OrangeTestBase.cc
//---------------------------------------------------------------------------//
#include "OrangeTestBase.hh"

#include "orange/OrangeData.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"

#include "geocel/CheckedGeoTrackView.t.hh"
#include "geocel/GenericGeoTestBase.t.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template class CheckedGeoTrackView<OrangeTrackView>;
template class GenericGeoTestBase<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
