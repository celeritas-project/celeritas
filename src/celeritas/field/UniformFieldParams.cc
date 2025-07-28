//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformFieldParams.cc
//---------------------------------------------------------------------------//
#include "UniformFieldParams.hh"

#include <unordered_set>
#include <utility>
#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/ext/GeantVolumeMapper.hh"

#include "UniformFieldData.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
std::unordered_set<VolumeId> make_volume_ids(VolumeParams const& volumes,
                                             GeantGeoParams const* geant_geo,
                                             inp::UniformField const& inp)
{
    using SetVolume = std::unordered_set<VolumeId>;

    return std::visit(
        Overload{[&](inp::UniformField::SetVolume const& s) {
                     CELER_EXPECT(geant_geo);
                     SetVolume result;
                     GeantVolumeMapper find_volume(*geant_geo);
                     for (auto const* lv : s)
                     {
                         CELER_ASSERT(lv);
                         auto vol = find_volume(*lv);
                         CELER_VALIDATE(vol < volumes.num_volumes(),
                                        << "failed to find volume while "
                                           "constructing a uniform field");
                         result.insert(vol);
                     }
                     return result;
                 },
                 [&](inp::UniformField::SetString const& s) {
                     SetVolume result;
                     for (auto const& name : s)
                     {
                         auto vols = volumes.volume_labels().find_all(name);
                         CELER_VALIDATE(!vols.empty(),
                                        << "failed to find volume '" << name
                                        << "' while constructing a uniform "
                                           "field");
                         result.insert(vols.begin(), vols.end());
                     }
                     return result;
                 },
                 [](std::monostate) { return SetVolume{}; }},
        inp.volumes);
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field.
 */
UniformFieldParams::UniformFieldParams(VolumeParams const& volume_params,
                                       SPConstGeantGeo geant_geo,
                                       Input const& inp)
{
    if (inp.units != UnitSystem::si)
    {
        CELER_NOT_IMPLEMENTED("field units in other unit systems");
    }

    HostVal<UniformFieldParamsData> host_data;

    // Interpret field strength in units of Tesla
    CELER_VALIDATE(norm(inp.strength) > 0,
                   << "along-step uniform field has zero field strength");
    for (auto i : range(inp.strength.size()))
    {
        host_data.field[i]
            = native_value_from(units::FieldTesla{inp.strength[i]});
    }

    // Throw a runtime error if any driver options are invalid
    validate_input(inp.driver_options);
    host_data.options = inp.driver_options;

    // If logical volumes are specified, flag whether or not the field should
    // be present in each volume
    auto volumes = make_volume_ids(volume_params, geant_geo.get(), inp);
    if (!volumes.empty())
    {
        std::vector<char> has_field(volume_params.num_volumes(), 0);
        for (auto vol : volumes)
        {
            CELER_VALIDATE(vol < volume_params.num_volumes(),
                           << "invalid volume ID "
                           << (vol ? vol.unchecked_get() : -1)
                           << " encountered while setting up uniform field");
            has_field[vol.unchecked_get()] = 1;
        }
        make_builder(&host_data.has_field)
            .insert_back(has_field.begin(), has_field.end());
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<UniformFieldParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
