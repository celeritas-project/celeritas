//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantGeoTestBase.hh"

#include "geocel/CheckedGeoTrackView.t.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GenericGeoTestBase.t.hh"
#include "geocel/g4/GeantGeoData.hh"
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
