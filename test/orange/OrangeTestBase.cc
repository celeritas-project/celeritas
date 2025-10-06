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
std::string OrangeTestBase::surface_name(GeoTrackView const& geo) const
{
    if (!geo.is_on_boundary())
    {
        return "---";
    }

    // Only call this function if the geometry supports surfaces
    return this->geometry()->surfaces().at(geo.impl_surface_id()).name;
}

//---------------------------------------------------------------------------//
template class CheckedGeoTrackView<OrangeTrackView>;
template class GenericGeoTestBase<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
