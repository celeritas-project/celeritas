//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/OpaqueId.hh"
#include "geocel/CheckedGeoTrackView.hh"
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
class OrangeTestBase : public GenericGeoTestBase<OrangeParams>
{
  public:
    std::string surface_name(GeoTrackView const& geo) const final;
};

extern template class CheckedGeoTrackView<OrangeTrackView>;
extern template class GenericGeoTestBase<OrangeParams>;

//---------------------------------------------------------------------------//
extern template class CheckedGeoTrackView<OrangeTrackView>;
extern template class GenericGeoTestBase<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
