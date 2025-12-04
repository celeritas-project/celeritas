//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Converter.cc
//---------------------------------------------------------------------------//
#include "Converter.hh"

#include <algorithm>
#include <fstream>
#include <variant>

#include "corecel/io/Logger.hh"
#include "geocel/GeantGeoParams.hh"
#include "geocel/Types.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/detail/LengthUnits.hh"
#include "orange/OrangeInput.hh"
#include "orange/OrangeTypes.hh"
#include "orange/orangeinp/InputBuilder.hh"

#include "PhysicalVolumeConverter.hh"
#include "ProtoConstructor.hh"

namespace celeritas
{
namespace g4org
{
namespace
{
//---------------------------------------------------------------------------//
bool is_null_volinst(VolumeInput const& vol)
{
    if (auto* vi_id = std::get_if<VolumeInstanceId>(&vol.label))
    {
        return *vi_id == VolumeInstanceId{};
    }
    return false;
}

//---------------------------------------------------------------------------//
/*!
 * Find the only volume that has a null volume instance label.
 */
LocalVolumeId find_bg_volume(std::vector<VolumeInput> const& volumes)
{
    auto iter = std::find_if(volumes.begin(), volumes.end(), is_null_volinst);
    CELER_ASSERT(iter != volumes.end());
    CELER_ASSERT(std::find_if(iter + 1, volumes.end(), is_null_volinst)
                 == volumes.end());
    return id_cast<LocalVolumeId>(iter - volumes.begin());
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with options.
 */
Converter::Converter(Options&& opts) : opts_{std::move(opts)}
{
    if (!opts_.tol)
    {
        opts_.tol = Tolerance<>::from_default(opts_.unit_length);
    }

    if (real_type{1} - ipow<2>(opts_.tol.rel) == real_type{1})
    {
        CELER_LOG(warning)
            << "Requested relative tolerance (" << opts_.tol.rel
            << ") for ORANGE is very small: tracking errors may result due to "
               "incomplete geometry simplification";
    }

    CELER_ENSURE(opts_.tol);
}

//---------------------------------------------------------------------------//
/*!
 * Convert the world.
 */
auto Converter::operator()(GeantGeoParams const& geo,
                           VolumeParams const& volumes) -> result_type
{
    using orangeinp::InputBuilder;

    // Convert solids, logical volumes, physical volumes
    PhysicalVolumeConverter convert_pv(geo, opts_);
    PhysicalVolume world = convert_pv(*geo.world());
    CELER_VALIDATE(std::holds_alternative<NoTransformation>(world.transform),
                   << "world volume should not have a transformation");

    // Convert logical volumes into protos
    auto global_proto = ProtoConstructor{volumes, opts_}(*world.lv);

    // Build universes from protos
    result_type result;
    InputBuilder build_input([&opts = opts_] {
        InputBuilder::Options ibo;
        ibo.tol = opts.tol;
        ibo.objects_output_file = opts.objects_output_file;
        ibo.csg_output_file = opts.csg_output_file;
        CELER_ENSURE(ibo);
        return ibo;
    }());
    result.input = build_input(*global_proto);

    // Replace the "background" (implicit *or* explicit) with the world volume
    // instance
    auto univ_iter = result.input.universes.begin();
    {
        // The first unit created is always the "world"; see detail::ProtoMap
        CELER_ASSERT(univ_iter != result.input.universes.end());
        CELER_ASSUME(std::holds_alternative<UnitInput>(*univ_iter));
        auto& unit = std::get<UnitInput>(*univ_iter++);

        // Find the only volume that has a null volume instance label
        LocalVolumeId bg_vol_id = find_bg_volume(unit.volumes);
        // Replace it with the world physical volume ID
        unit.volumes[bg_vol_id.get()].label = world.id;
        // Do *not* set the 'background' field for it, since it truly
        // represents a volume instance

        // Replace any null targets with the world PV
        for (auto& [src, tgt] : unit.local_parent_map)
        {
            CELER_ASSERT(src);
            if (!tgt)
            {
                tgt = bg_vol_id;
            }
        }
    }
    // Replace other backgrounds, annotating with the corresponding volume
    // (note it's not a volume instance!)
    for (; univ_iter != result.input.universes.end(); ++univ_iter)
    {
        if (auto* unit = std::get_if<UnitInput>(&(*univ_iter)))
        {
            // Find the only volume that has a null volume instance label
            LocalVolumeId bg_vol_id = find_bg_volume(unit->volumes);
            // Save the "implementation volume" name, and annotate the
            // corresponding volume ID
            unit->volumes[bg_vol_id.get()].label.emplace<Label>(
                "[BG]", unit->label.name);
            unit->background.label
                = volumes.volume_labels().find_exact(unit->label);
            unit->background.volume = bg_vol_id;
        }
    }

    if (!opts_.org_output_file.empty())
    {
        CELER_LOG(info) << "Writing constructed ORANGE geometry to "
                        << opts_.org_output_file;
        // Export constructed geometry for debugging
        std::ofstream outf(opts_.org_output_file);
        CELER_VALIDATE(outf,
                       << "failed to open output file at \""
                       << opts_.org_output_file << '"');
        outf << result.input;
    }

    return result;
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
