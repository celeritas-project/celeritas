//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantSurfacePhysicsHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <G4LogicalSurface.hh>
#include <G4OpticalSurface.hh>

#include "corecel/inp/Grid.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/Types.hh"

#include "GeantMaterialPropertyGetter.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Helper class used by \c GeantSurfacePhysicsLoader .
 */
class GeantSurfacePhysicsHelper
{
  public:
    // Construct with SurfaceId; this expects a valid GeantGeoParams
    GeantSurfacePhysicsHelper(SurfaceId sid);

    //! True if this helper is fully constructed and valid
    explicit operator bool() const { return surface_ != nullptr; }

    // Get optical surface id
    SurfaceId surface_id() const { return sid_; }

    // Get Geant4 optical surface
    G4OpticalSurface const& surface() const;

    // Access the raw property getter for this surface
    GeantMaterialPropertyGetter const& property_getter() const
    {
        CELER_ASSERT(*this);
        return get_property_;
    }

    // Populate Grid optical property from name, in [MeV, unitless]
    bool get_property(inp::Grid& dst, std::string const& name) const;

  private:
    SurfaceId sid_;
    G4OpticalSurface const* surface_;
    GeantMaterialPropertyGetter get_property_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
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
    if (surface_)
    {
        // The surface may not be an optical surface, or it might have no
        // properties
        get_property_ = GeantMaterialPropertyGetter{
            surface_->GetMaterialPropertiesTable(), surface_->GetName()};
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get Geant4 optical surface.
 */
G4OpticalSurface const& GeantSurfacePhysicsHelper::surface() const
{
    CELER_ASSERT(*this);
    return *surface_;
}

//---------------------------------------------------------------------------//
/*!
 * Get mev-to-unitless grid property from material properties table.
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
bool GeantSurfacePhysicsHelper::get_property(inp::Grid& dst,
                                             std::string const& name) const
{
    auto loaded
        = get_property_(dst, name, {ImportUnits::mev, ImportUnits::unitless});
    return loaded;
}

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
//! Write surface name and surface id to a stream
inline std::ostream&
operator<<(std::ostream& os, GeantSurfacePhysicsHelper const& helper)
{
    os << helper.property_getter()
       << " (surface id=" << helper.surface_id().unchecked_get() << ")";
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
