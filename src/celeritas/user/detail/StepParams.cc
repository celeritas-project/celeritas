//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepParams.cc
//---------------------------------------------------------------------------//
#include "StepParams.hh"

#include "corecel/io/Label.hh"
#include "geocel/VolumeCollectionBuilder.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/geo/CoreGeoParams.hh"

#include "../StepInterface.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct from data IDs and interfaces.
 */
StepParams::StepParams(AuxId aux_id,
                       CoreGeoParams const& geo,
                       VecInterface const& callbacks)
    : aux_id_{aux_id}
{
    CELER_EXPECT(aux_id_);

    enum class HasDetectors
    {
        unknown = -1,
        none,
        all
    };

    // FIXME: pass these into the constructor rather than using globals
    auto const& volume_params = celeritas::global_volumes().lock();
    CELER_ASSERT(volume_params);

    auto const& volumes = volume_params->volume_labels();
    StepSelection selection;
    CELER_ASSERT(!selection);
    StepInterface::MapVolumeDetector detector_map;
    bool nonzero_energy_deposition{true};

    // Loop over callbacks to take union of step selections
    HasDetectors has_det = HasDetectors::unknown;

    for (SPStepInterface const& sp_interface : callbacks)
    {
        auto&& this_selection = sp_interface->selection();
        CELER_VALIDATE(this_selection,
                       << "step interface doesn't collect any data");
        selection |= this_selection;

        auto const&& filters = sp_interface->filters();
        for (auto const& kv : filters.detectors)
        {
            // Map detector volumes, asserting uniqueness
            CELER_ASSERT(kv.first);
            auto [prev, inserted] = detector_map.insert(kv);
            CELER_VALIDATE(
                inserted,
                << "a single volume is assigned to multiple detectors ('"
                << volumes.at(prev->first) << "' -> " << prev->second.get()
                << " and '" << volumes.at(kv.first) << "' -> "
                << kv.second.get() << ')');
        }

        // Filter out zero-energy steps/tracks only if all detectors agree
        nonzero_energy_deposition = nonzero_energy_deposition
                                    && filters.nonzero_energy_deposition;

        auto this_has_detectors = filters.detectors.empty()
                                      ? HasDetectors::none
                                      : HasDetectors::all;
        if (has_det == HasDetectors::unknown)
        {
            has_det = this_has_detectors;
        }
        CELER_VALIDATE(this_has_detectors == has_det,
                       << "inconsistent step callbacks: mixing those with "
                          "detectors and those without is currently "
                          "unsupported");
    }
    CELER_ASSERT(selection);
    CELER_ASSERT((has_det == HasDetectors::none) == detector_map.empty());

    auto vol_to_det = [&detector_map](VolumeId vol_id) {
        if (auto iter = detector_map.find(vol_id); iter != detector_map.end())
        {
            // The real volume maps to a detector
            return iter->second;
        }
        return DetectorId{};
    };

    // Save data
    mirror_ = ParamsDataStore{[&] {
        HostVal<StepParamsData> host_data;
        host_data.selection = selection;

        if (!detector_map.empty())
        {
            host_data.detector
                = build_volume_collection<DetectorId>(geo, vol_to_det);
            host_data.nonzero_energy_deposition = nonzero_energy_deposition;
            CELER_ASSERT(!host_data.detector.empty());
        }

        if (selection.points[StepPoint::pre].volume_instance_ids
            || selection.points[StepPoint::post].volume_instance_ids)
        {
            // TODO: replace with volume params, so we can use touchable
            // representation
            host_data.num_volume_levels = volume_params->num_volume_levels();
            CELER_VALIDATE(host_data.num_volume_levels > 0,
                           << "geometry type does not support volume "
                              "instance IDs: max depth is "
                           << host_data.num_volume_levels);
        }

        return host_data;
    }()};

    CELER_ASSERT((has_det == HasDetectors::all) == this->has_detectors());
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
