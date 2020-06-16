//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGHost.cc
//---------------------------------------------------------------------------//
#include "VGHost.hh"

#include <iostream>

#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include <celeritas_config.h>
#ifdef CELERITAS_USE_CUDA
#    include <VecGeom/management/CudaManager.h>
#endif

#include "VGView.hh"

using std::cout;
using std::endl;

namespace celeritas
{
//---------------------------------------------------------------------------//
// MANAGEMENT
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
VGHost::VGHost(const char* gdml_filename)
{
    cout << "::: Loading from GDML at " << gdml_filename << endl;
    // NOTE: the validation check disabling is missing from vecgeom 1.1.6 and
    // earlier; without it, the VGDML loader may crash.
    constexpr bool validate_xml_schema = false;
    vgdml::Frontend::Load(gdml_filename, validate_xml_schema);
    cout << "::: Initializing tracking information" << endl;
    vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();

    num_volumes_ = vecgeom::VPlacedVolume::GetIdCount();
    max_depth_   = vecgeom::GeoManager::Instance().getMaxDepth();
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a placed volume ID
 */
const std::string& VGHost::id_to_label(VolumeId vol_id) const
{
    REQUIRE(vol_id.get() < num_volumes_);
    const auto* vol
        = vecgeom::GeoManager::Instance().FindPlacedVolume(vol_id.get());
    CHECK(vol);
    return vol->GetLabel();
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label
 */
auto VGHost::label_to_id(const std::string& label) const -> VolumeId
{
    const auto* vol
        = vecgeom::GeoManager::Instance().FindPlacedVolume(label.c_str());
    CHECK(vol);
    CHECK(vol->id() < num_volumes_);
    return VolumeId{vol->id()};
}

//---------------------------------------------------------------------------//
/*!
 * View in-host geometry data for CPU debugging.
 */
VGView VGHost::host_view() const
{
    VGView result;
    result.world_volume = vecgeom::GeoManager::Instance().GetWorld();
    return result;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
