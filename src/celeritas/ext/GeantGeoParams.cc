//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantGeoParams.cc
//---------------------------------------------------------------------------//
#include "GeantGeoParams.hh"

#include <vector>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4Transportation.hh>
#include <G4TransportationManager.hh>
#include <G4VSolid.hh>
#include <G4Version.hh>
#include <G4VisExtent.hh>
#if G4VERSION_NUMBER >= 1070
#    include <G4Backtrace.hh>
#endif

#include "celeritas_config.h"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedMem.hh"

#include "Convert.geant.hh"  // IWYU pragma: associated
#include "GeantGeoData.hh"  // IWYU pragma: associated
#include "GeantGeoUtils.hh"  // IWYU pragma: associated
#include "ScopedGeantExceptionHandler.hh"
#include "detail/GeantLoggerAdapter.hh"
#include "detail/GeantVolumeVisitor.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 *
 * This assumes that Celeritas is driving and will manage Geant4 exceptions
 * etc.
 */
GeantGeoParams::GeantGeoParams(std::string const& filename)
{
    ScopedMem record_mem("GeantGeoParams.construct");

#if G4VERSION_NUMBER >= 1070
    // Disable geant4 signal interception
    G4Backtrace::DefaultSignals() = {};
#endif

    detail::GeantLoggerAdapter scoped_logger;
    scoped_exceptions_ = std::make_unique<ScopedGeantExceptionHandler>();

    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    host_ref_.world = load_geant_geometry(filename);
    loaded_gdml_ = true;

    this->build_tracking();
    this->build_metadata();

    CELER_ENSURE(this->num_volumes() > 0);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Use an existing loaded Geant4 geometry.
 */
GeantGeoParams::GeantGeoParams(G4VPhysicalVolume const* world)
{
    CELER_EXPECT(world);
    host_ref_.world = const_cast<G4VPhysicalVolume*>(world);

    ScopedMem record_mem("GeantGeoParams.construct");

    // Verify consistency of the world volume
    G4VPhysicalVolume const* nav_world = [] {
        auto* man = G4TransportationManager::GetTransportationManager();
        CELER_ASSERT(man);
        auto* nav = man->GetNavigatorForTracking();
        CELER_ENSURE(nav);
        return nav->GetWorldVolume();
    }();
    if (world != nav_world)
    {
        auto msg = CELER_LOG(warning);
        msg << "Geant4 geometry was initialized with inconsistent "
               "world volume: given '"
            << world->GetName() << "'@' " << static_cast<void const*>(world)
            << "; navigation world is ";
        if (nav_world)
        {
            msg << nav_world->GetName() << "'@' "
                << static_cast<void const*>(nav_world);
        }
        else
        {
            msg << "unset";
        }
    }

    this->build_tracking();
    this->build_metadata();

    CELER_ENSURE(this->num_volumes() > 0);
    CELER_ENSURE(host_ref_);
}

//---------------------------------------------------------------------------//
/*!
 * Clean up on destruction.
 */
GeantGeoParams::~GeantGeoParams()
{
    if (closed_geometry_)
    {
        G4GeometryManager::GetInstance()->OpenGeometry(host_ref_.world);
    }
    if (loaded_gdml_)
    {
        reset_geant_geometry();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a placed volume ID.
 */
Label const& GeantGeoParams::id_to_label(VolumeId vol) const
{
    CELER_EXPECT(vol < vol_labels_.size());
    return vol_labels_.get(vol);
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label.
 */
auto GeantGeoParams::find_volume(std::string const& name) const -> VolumeId
{
    auto result = vol_labels_.find_all(name);
    if (result.empty())
        return {};
    CELER_VALIDATE(result.size() == 1,
                   << "volume '" << name << "' is not unique");
    return result.front();
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a label.
 *
 * If the label isn't in the geometry, a null ID will be returned.
 */
VolumeId GeantGeoParams::find_volume(Label const& label) const
{
    return vol_labels_.find(label);
}

//---------------------------------------------------------------------------//
/*!
 * Locate the volume ID corresponding to a Geant4 logical volume.
 */
VolumeId GeantGeoParams::find_volume(G4LogicalVolume const* volume) const
{
    CELER_EXPECT(volume);
    auto inst_id = volume->GetInstanceID();
    CELER_ENSURE(inst_id >= 0);
    return VolumeId{static_cast<VolumeId::size_type>(inst_id)};
}

//---------------------------------------------------------------------------//
/*!
 * Get zero or more volume IDs corresponding to a name.
 *
 * This is useful for volumes that are repeated in the geometry with different
 * uniquifying 'extensions' from Geant4.
 */
auto GeantGeoParams::find_volumes(std::string const& name) const
    -> SpanConstVolumeId
{
    return vol_labels_.find_all(name);
}

//---------------------------------------------------------------------------//
/*!
 * Complete geometry construction
 */
void GeantGeoParams::build_tracking()
{
    // Close the geometry if needed
    auto* geo_man = G4GeometryManager::GetInstance();
    CELER_ASSERT(geo_man);
    if (!geo_man->IsGeometryClosed())
    {
        geo_man->CloseGeometry(
            /* optimize = */ true, /* verbose = */ false, host_ref_.world);
        closed_geometry_ = true;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct Celeritas host-only metadata.
 */
void GeantGeoParams::build_metadata()
{
    CELER_EXPECT(host_ref_);

    ScopedMem record_mem("GeantGeoParams.build_metadata");
    // Construct volume labels
    vol_labels_ = LabelIdMultiMap<VolumeId>(
        [world = host_ref_.world, unique_volumes = !loaded_gdml_] {
            G4LogicalVolume const* lv = world->GetLogicalVolume();
            CELER_ASSERT(lv);

            // Recursive loop over all logical volumes to populate map
            detail::GeantVolumeVisitor visitor(unique_volumes);
            visitor.visit(*lv);
            return visitor.build_label_vector();
        }());

    // Save world bbox (NOTE: assumes no transformation on PV?)
    bbox_ = [world = host_ref_.world] {
        G4LogicalVolume const* lv = world->GetLogicalVolume();
        CELER_ASSERT(lv);
        G4VSolid const* solid = lv->GetSolid();
        CELER_ASSERT(solid);

        G4VisExtent const& extent = solid->GetExtent();

        return BoundingBox({convert_from_geant(extent.GetXmin(), CLHEP::cm),
                            convert_from_geant(extent.GetYmin(), CLHEP::cm),
                            convert_from_geant(extent.GetZmin(), CLHEP::cm)},
                           {convert_from_geant(extent.GetXmax(), CLHEP::cm),
                            convert_from_geant(extent.GetYmax(), CLHEP::cm),
                            convert_from_geant(extent.GetZmax(), CLHEP::cm)});
    }();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
