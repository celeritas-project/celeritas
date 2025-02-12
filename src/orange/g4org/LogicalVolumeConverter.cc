//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/LogicalVolumeConverter.cc
//---------------------------------------------------------------------------//
#include "LogicalVolumeConverter.hh"

#include <G4LogicalVolume.hh>
#include <G4Material.hh>
#include <G4VSolid.hh>

#include "corecel/Assert.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Environment.hh"
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
LogicalVolumeConverter::LogicalVolumeConverter(VecLabel const& labels,
                                               SolidConverter& convert_solid)
    : labels_{labels}, convert_solid_(convert_solid)
{
}

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 logical volume to an ORANGE LogicalVolume.
 *
 * This uses a cache to look up any previously converted volume.
 */
auto LogicalVolumeConverter::operator()(arg_type lv) -> result_type
{
    SPLV result;
    auto [cache_iter, inserted] = cache_.insert({&lv, {}});
    if (inserted || cache_iter->second.expired())
    {
        // First time converting the volume, or it has expired
        result = this->construct_impl(lv);
        cache_iter->second = result;
    }
    else
    {
        result = cache_iter->second.lock();
    }

    CELER_ENSURE(result);
    return {result, inserted};
}

//---------------------------------------------------------------------------//
/*!
 * Convert the raw logical volume without processing any daughters.
 */
auto LogicalVolumeConverter::construct_impl(arg_type g4lv) -> SPLV
{
    auto result = std::make_shared<LogicalVolume>();

    // Save Geant4 volume pointer for later mappings
    result->g4lv = &g4lv;

    // Save name
    result->label = [&] {
        Label result;
        if (static_cast<std::size_t>(g4lv.GetInstanceID()) < labels_.size())
        {
            result = labels_[g4lv.GetInstanceID()];
        }
        if (result.empty())
        {
            result.name = g4lv.GetName();
            CELER_LOG(warning) << "Logical volume '" << result.name
                               << "' is not in the world geometry";
        }
        return result;
    }();

    // Save material ID
    // NOTE: this is *not* the physics material ("cut couple")
    if (auto* mat = g4lv.GetMaterial())
    {
        result->material_id = id_cast<GeoMaterialId>(mat->GetIndex());
    }
    else
    {
        CELER_LOG(warning) << "Logical volume '" << result->label
                           << "' has no associated material";
        result->material_id = GeoMaterialId{0};
    }

    // Convert solid
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
    catch (celeritas::DebugError const& e)
    {
        CELER_LOG(error) << "Failed to convert solid type '"
                         << g4lv.GetSolid()->GetEntityType() << "' named '"
                         << g4lv.GetSolid()->GetName();
        CELER_LOG(info) << "Unsupported solid belongs to logical volume "
                        << PrintableLV{&g4lv};

        static bool const allow_errors
            = !celeritas::getenv("G4ORG_ALLOW_ERRORS").empty();
        if (!allow_errors)
        {
            throw;
        }
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
