//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4vg/LogicalVolumeConverter.cc
//---------------------------------------------------------------------------//
#include "LogicalVolumeConverter.hh"

#include <G4LogicalVolume.hh>
#include <G4VSolid.hh>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/LogicalVolume.h>
#include <VecGeom/volumes/UnplacedVolume.h>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoUtils.hh"

#include "SolidConverter.hh"

namespace celeritas
{
namespace g4vg
{
//---------------------------------------------------------------------------//
/*!
 * Construct with solid conversion helper.
 */
LogicalVolumeConverter::LogicalVolumeConverter(SolidConverter& convert_solid)
    : convert_solid_(convert_solid)
{
    CELER_EXPECT(!vecgeom::GeoManager::Instance().IsClosed());
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 logical volume to a VecGeom LogicalVolume.
 *
 * This uses a cache to look up any previously converted volume.
 */
auto LogicalVolumeConverter::operator()(arg_type lv) -> result_type
{
    auto [cache_iter, inserted] = cache_.insert({&lv, nullptr});
    if (inserted)
    {
        // First time converting the volume
        cache_iter->second = this->construct_base(lv);
    }

    CELER_ENSURE(cache_iter->second);
    return cache_iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a mapping from G4 logical volume to VolumeId.
 */
auto LogicalVolumeConverter::make_volume_map() const -> MapLvVolId
{
    MapLvVolId result;
    result.reserve(cache_.size());

    for (auto const& kv : cache_)
    {
        CELER_ASSERT(kv.second);
        result.insert({kv.first, VolumeId{kv.second->id()}});
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Convert the raw logical volume from geant4 to vecgeom.
 */
auto LogicalVolumeConverter::construct_base(arg_type g4lv) -> result_type
{
    vecgeom::VUnplacedVolume const* shape = nullptr;
    try
    {
        shape = convert_solid_(*g4lv.GetSolid());
    }
    catch (celeritas::RuntimeError const& e)
    {
        CELER_LOG(error) << "Failed to convert solid type '"
                         << g4lv.GetSolid()->GetEntityType() << "' named '"
                         << g4lv.GetSolid()->GetName()
                         << "': " << e.details().what;
        shape = this->convert_solid_.to_sphere(*g4lv.GetSolid());
        CELER_LOG(warning)
            << "Replaced unknown solid with sphere with capacity "
            << shape->Capacity() << " [len^3]";
        CELER_LOG(info) << "Unsupported solid belongs to logical volume "
                        << PrintableLV{&g4lv};
    }

    std::string name = g4lv.GetName();
    if (name.find("0x") == std::string::npos)
    {
        // No pointer address: add one
        name = make_gdml_name(g4lv);
    }

    return new vecgeom::LogicalVolume(name.c_str(), shape);
}

//---------------------------------------------------------------------------//
}  // namespace g4vg
}  // namespace celeritas
