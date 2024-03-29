//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/LogicalVolumeConverter.cc
//---------------------------------------------------------------------------//
#include "LogicalVolumeConverter.hh"

#include <G4LogicalVolume.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4VSolid.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoUtils.hh"

#include "SolidConverter.hh"
#include "Volume.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Construct with solid conversion helper.
 */
LogicalVolumeConverter::LogicalVolumeConverter(SolidConverter& convert_solid)
    : convert_solid_(convert_solid)
{
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
        cache_iter->second = this->construct_impl(lv);
    }

    CELER_ENSURE(cache_iter->second);
    return cache_iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Convert the raw logical volume without processing any daughters.
 */
auto LogicalVolumeConverter::construct_impl(arg_type g4lv) -> result_type
{
    auto result = std::make_shared<LogicalVolume>();
    result->g4lv = &g4lv;

    try
    {
        result->solid = convert_solid_(*g4lv.GetSolid());
    }
    catch (celeritas::RuntimeError const& e)
    {
        CELER_LOG(error) << "Failed to convert solid type '"
                         << g4lv.GetSolid()->GetEntityType() << "' named '"
                         << g4lv.GetSolid()->GetName()
                         << "': " << e.details().what;
        result->solid = this->convert_solid_.to_sphere(*g4lv.GetSolid());
        CELER_LOG(warning) << "Replaced unknown solid with sphere ("
                           << to_string(*result->solid) << ")";
        CELER_LOG(info) << "Unsupported solid belongs to logical volume "
                        << PrintableLV{&g4lv};
    }

    // Save material ID
    if (auto* cuts = g4lv.GetMaterialCutsCouple())
    {
        auto idx = cuts->GetIndex();
        CELER_ASSERT(idx >= 0);
        result->material_id = MaterialId{static_cast<size_type>(idx)};
    }

    // Save name
    result->name = g4lv.GetName();
    if (result->name.find("0x") == std::string::npos)
    {
        // No pointer address: add one
        result->name = make_gdml_name(g4lv);
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
