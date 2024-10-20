//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4/GeantGeoParams.cc
//---------------------------------------------------------------------------//
#include "GeantGeoParams.hh"

#include <vector>
#include <G4GeometryManager.hh>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4Transportation.hh>
#include <G4TransportationManager.hh>
#include <G4VSolid.hh>
#include <G4Version.hh>
#include <G4VisExtent.hh>
#if G4VERSION_NUMBER >= 1070
#    include <G4Backtrace.hh>
#endif

#include "corecel/Config.hh"

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ScopedMem.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"

#include "Convert.geant.hh"  // IWYU pragma: associated
#include "GeantGeoData.hh"  // IWYU pragma: associated
#include "VisitGeantVolumes.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
std::vector<Label>
get_volume_labels(G4LogicalVolume const& world, bool unique_volumes)
{
    std::vector<Label> labels;
    visit_geant_volumes(
        [&](G4LogicalVolume const& lv) {
            auto i = static_cast<std::size_t>(lv.GetInstanceID());
            if (i >= labels.size())
            {
                labels.resize(i + 1);
            }
            if (unique_volumes)
            {
                labels[i] = Label::from_geant(make_gdml_name(lv));
            }
            else
            {
                labels[i] = Label::from_geant(lv.GetName());
            }
        },
        world);
    return labels;
}

//---------------------------------------------------------------------------//
}  // namespace

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

    disable_geant_signal_handler();

    if (!ends_with(filename, ".gdml"))
    {
        CELER_LOG(warning) << "Expected '.gdml' extension for GDML input";
    }

    host_ref_.world = load_geant_geometry(filename);
    loaded_gdml_ = true;

    // NOTE: only instantiate the logger/exception handler *after* loading
    // Geant4 geometry, since something in the GDML parser's call chain resets
    // the Geant logger.
    scoped_logger_ = std::make_unique<ScopedGeantLogger>();
    scoped_exceptions_ = std::make_unique<ScopedGeantExceptionHandler>();

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
    auto result = id_cast<VolumeId>(volume->GetInstanceID());
    if (!(result < this->num_volumes()))
    {
        // Volume is out of range: possibly an LV defined after this geometry
        // class was created
        result = {};
    }
    return result;
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
 * Get the Geant4 logical volume corresponding to a volume ID.
 *
 * If the input volume ID is false, a null pointer will be returned.
 */
G4LogicalVolume const* GeantGeoParams::id_to_lv(VolumeId id) const
{
    CELER_EXPECT(!id || id < this->num_volumes());
    if (!id)
    {
        return nullptr;
    }

    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    CELER_ASSERT(id < lv_store->size());
    return (*lv_store)[id.unchecked_get()];
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

    auto const* world_lv = host_ref_.world->GetLogicalVolume();
    CELER_ASSERT(world_lv);

    // Construct volume labels
    vol_labels_ = LabelIdMultiMap<VolumeId>(
        get_volume_labels(*world_lv, !loaded_gdml_));

    // Save world bbox (NOTE: assumes no transformation on PV?)
    bbox_ = [world_lv] {
        G4VSolid const* solid = world_lv->GetSolid();
        CELER_ASSERT(solid);
        G4VisExtent const& extent = solid->GetExtent();

        return BBox({convert_from_geant(G4ThreeVector(extent.GetXmin(),
                                                      extent.GetYmin(),
                                                      extent.GetZmin()),
                                        clhep_length),
                     convert_from_geant(G4ThreeVector(extent.GetXmax(),
                                                      extent.GetYmax(),
                                                      extent.GetZmax()),
                                        clhep_length)});
    }();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
