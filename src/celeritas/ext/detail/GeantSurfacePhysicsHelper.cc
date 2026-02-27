//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsHelper.cc
//---------------------------------------------------------------------------//
#include "GeantSurfacePhysicsHelper.hh"

#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>

#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"

#include "GeantMaterialPropertyGetter.hh"

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
    auto geo = celeritas::global_geant_geo().lock();
    CELER_ASSERT(geo);
    auto const* g4log_surf = geo->id_to_geant(sid);
    CELER_ASSERT(g4log_surf);
    auto* g4surf_prop = g4log_surf->GetSurfaceProperty();
    CELER_ASSERT(g4surf_prop);
    surface_ = dynamic_cast<G4OpticalSurface*>(g4surf_prop);
    CELER_ASSERT(surface_);
    mpt_ = surface_->GetMaterialPropertiesTable();
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
 *
 * \note Currently all imported parameters are in [energy] vs. [unitless], and
 * therefore units are abstracted from the function call. The grids currently
 * pulled by this helper are:
 * - Reflectivity
 * - Transmittance
 * - Efficiency
 * - Specular spike
 * - Specular lobe
 * - Backscatter
 * - Surface refractive index
 */
bool GeantSurfacePhysicsHelper::get_property(inp::Grid* dst,
                                             std::string const& name) const
{
    if (!mpt_)
    {
        return false;
    }

    GeantMaterialPropertyGetter get_property{*mpt_};
    auto loaded
        = get_property(dst, name, {ImportUnits::mev, ImportUnits::unitless});
    if (loaded)
    {
        CELER_LOG(debug) << "Loaded " << name << " from "
                         << this->surface().GetName();
    }
    return loaded;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
