//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantGeoTestBase.hh"

#include "celeritas/CheckedGeoTrackView.t.hh"
#include "celeritas/GenericGeoTestBase.t.hh"
#include "geocel/g4/GeantGeoData.hh"
#include "geocel/g4/GeantGeoParams.hh"
#include "geocel/g4/GeantGeoTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template class CheckedGeoTrackView<GeantGeoTrackView>;
template class GenericGeoTestBase<GeantGeoParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
