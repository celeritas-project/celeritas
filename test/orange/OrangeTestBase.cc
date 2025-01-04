//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTestBase.cc
//---------------------------------------------------------------------------//
#include "OrangeTestBase.hh"

#include "geocel/CheckedGeoTrackView.t.hh"
#include "geocel/GenericGeoTestBase.t.hh"
#include "orange/OrangeData.hh"
#include "orange/OrangeParams.hh"
#include "orange/OrangeTrackView.hh"

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
