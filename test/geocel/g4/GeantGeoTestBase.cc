//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoTestBase.cc
//---------------------------------------------------------------------------//
#include "GeantGeoTestBase.hh"

#include "geocel/GenericGeoTestBase.t.hh"  // IWYU pragma: keep
#include "geocel/g4/GeantGeoTrackView.hh"  // IWYU pragma: keep

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template class GenericGeoTestBase<GeantGeoParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
