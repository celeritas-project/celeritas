//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/VecgeomTestBase.cc
//---------------------------------------------------------------------------//
#include "VecgeomTestBase.hh"

#include "celeritas/CheckedGeoTrackView.t.hh"
#include "celeritas/GenericGeoTestBase.t.hh"
#include "celeritas/ext/VecgeomData.hh"
#include "celeritas/ext/VecgeomParams.hh"
#include "celeritas/ext/VecgeomTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template class CheckedGeoTrackView<VecgeomTrackView>;
template class GenericGeoTestBase<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
