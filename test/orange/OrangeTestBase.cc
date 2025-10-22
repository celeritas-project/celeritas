//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeTestBase.cc
//---------------------------------------------------------------------------//
#include "OrangeTestBase.hh"

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

    auto& wrapped = dynamic_cast<WrappedGeoTrack const&>(geo);
    auto impl_surface = wrapped.track_view().impl_surface_id();
    return this->geometry()->surfaces().at(impl_surface).name;
}

//---------------------------------------------------------------------------//
template class GenericGeoTestBase<OrangeParams>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
