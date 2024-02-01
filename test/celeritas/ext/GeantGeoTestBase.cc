//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/GeantGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantGeoTestBase.hh"

#include "celeritas/CheckedGeoTrackView.t.hh"
#include "celeritas/GenericGeoTestBase.t.hh"
#include "celeritas/ext/GeantGeoData.hh"
#include "celeritas/ext/GeantGeoParams.hh"
#include "celeritas/ext/GeantGeoTrackView.hh"

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
