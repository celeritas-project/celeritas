//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/VecgeomTestBase.cc
//---------------------------------------------------------------------------//
#include "VecgeomTestBase.hh"

#include "geocel/GenericGeoTestBase.t.hh"
#include "geocel/vg/VecgeomData.hh"
#include "geocel/vg/VecgeomParams.hh"
#include "geocel/vg/VecgeomTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template class GenericGeoTestBase<VecgeomParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
