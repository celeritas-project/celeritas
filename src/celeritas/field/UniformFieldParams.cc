//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformFieldParams.cc
//---------------------------------------------------------------------------//
#include "UniformFieldParams.hh"

#include <unordered_set>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/VariantUtils.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/VolumeCollectionBuilder.hh"
#include "geocel/VolumeIdBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/inp/Field.hh"

#include "UniformFieldData.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
std::unordered_set<VolumeId> make_volume_ids(inp::UniformField const& inp)
{
    using SetVolume = std::unordered_set<VolumeId>;

    VolumeIdBuilder to_vol_id;
    SetVolume result;

    std::visit(Overload{
                   [](std::monostate) { /* no volumes provided */ },
                   [&](auto const& set_labels) {
                       for (auto&& lv_or_str : set_labels)
                       {
                           result.insert(to_vol_id(lv_or_str));
                       }
                   },
               },
               inp.volumes);
    CELER_VALIDATE(std::all_of(result.begin(), result.end(), Identity{}),
                   << "failed to find one or more volumes while "
                      "constructing a uniform field");
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct from a user-defined field.
 */
UniformFieldParams::UniformFieldParams(CoreGeoParams const& geo,
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
    auto volumes = make_volume_ids(inp);
    if (!volumes.empty())
    {
        // Convert from canonical to implementation volumes
        host_data.has_field
            = build_volume_collection<char>(geo, [&volumes](VolumeId vid) {
                  return static_cast<bool>(volumes.count(vid));
              });
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<UniformFieldParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
