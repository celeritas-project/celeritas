//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsHelper.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsHelper.hh"

#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>

#include "geocel/GeantGeoParams.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with valid SurfaceId. A GeantGeoParams in a valid state is also
 * required.
 */
GeantSurfacePhysicsHelper::GeantSurfacePhysicsHelper(SurfaceId sid) : sid_(sid)
{
    CELER_EXPECT(sid_);
    auto geo = celeritas::geant_geo().lock();
    CELER_ASSERT(geo);
    auto const* g4log_surf = geo->id_to_geant(sid);
    CELER_ASSERT(g4log_surf);
    auto* g4surf_prop = g4log_surf->GetSurfaceProperty();
    CELER_ASSERT(g4surf_prop);
    surface_ = dynamic_cast<G4OpticalSurface*>(g4surf_prop);
    CELER_ASSERT(surface_);
    mpt_ = surface_->GetMaterialPropertiesTable();
    CELER_ASSERT(mpt_);
}

//---------------------------------------------------------------------------//
/*!
 * Get Geant4 optical surface.
 */
G4OpticalSurface const& GeantSurfacePhysicsHelper::surface() const
{
    return *surface_;
}

//---------------------------------------------------------------------------//
/*!
 * Get property from material properties table.
 */
bool GeantSurfacePhysicsHelper::get_property(inp::Grid* dst,
                                             std::string const& name)
{
    GeantMaterialPropertyGetter get_property{*mpt_};
    return get_property(dst, name, {ImportUnits::mev, ImportUnits::unitless});
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
