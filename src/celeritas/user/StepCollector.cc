//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepCollector.cc
//---------------------------------------------------------------------------//
#include "StepCollector.hh"

#include <algorithm>
#include <map>
#include <type_traits>
#include <utility>

#include "corecel/cont/EnumArray.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/io/Label.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/user/StepInterface.hh"
#include "celeritas/user/detail/StepStorage.hh"

#include "detail/StepGatherAction.hh"

namespace celeritas
{
namespace
{
enum class HasDetectors
{
    unknown = -1,
    none,
    all
};
}

//---------------------------------------------------------------------------//
/*!
 * Construct with options and register pre and/or post-step actions.
 */
StepCollector::StepCollector(VecInterface callbacks,
                             SPConstGeo geo,
                             size_type num_streams,
                             ActionRegistry* action_registry)
    : storage_(std::make_shared<detail::StepStorage>())
{
    CELER_EXPECT(!callbacks.empty());
    CELER_EXPECT(std::all_of(
        callbacks.begin(), callbacks.end(), [](SPStepInterface const& i) {
            return static_cast<bool>(i);
        }));
    CELER_EXPECT(geo);
    CELER_EXPECT(num_streams > 0);
    CELER_EXPECT(action_registry);

    // Loop over callbacks to take union of step selections
    StepSelection selection;
    StepInterface::MapVolumeDetector detector_map;
    bool nonzero_energy_deposition{true};
    {
        CELER_ASSERT(!selection);

        HasDetectors has_detectors = HasDetectors::unknown;

        for (SPStepInterface const& sp_interface : callbacks)
        {
            auto this_selection = sp_interface->selection();
            CELER_VALIDATE(this_selection,
                           << "step interface doesn't collect any data");
            selection |= this_selection;

            auto const&& filters = sp_interface->filters();
            for (auto const& kv : filters.detectors)
            {
                // Map detector volumes, asserting uniqueness
                CELER_ASSERT(kv.first);
                auto [prev, inserted] = detector_map.insert(kv);
                CELER_VALIDATE(inserted,
                               << "multiple step interfaces map single volume "
                                  "to a detector ('"
                               << geo->id_to_label(prev->first) << "' -> "
                               << prev->second.get() << " and '"
                               << geo->id_to_label(kv.first) << "' -> "
                               << kv.second.get() << ')');
            }

            // Filter out zero-energy steps/tracks only if all detectors agree
            nonzero_energy_deposition = nonzero_energy_deposition
                                        && filters.nonzero_energy_deposition;

            auto this_has_detectors = filters.detectors.empty()
                                          ? HasDetectors::none
                                          : HasDetectors::all;
            if (has_detectors == HasDetectors::unknown)
            {
                has_detectors = this_has_detectors;
            }
            CELER_VALIDATE(this_has_detectors == has_detectors,
                           << "inconsistent step callbacks: mixing those with "
                              "detectors and those without is currently "
                              "unsupported");
        }
        CELER_ASSERT(selection);
    }

    {
        // Create params
        celeritas::HostVal<StepParamsData> host_data;

        host_data.selection = selection;

        if (!detector_map.empty())
        {
            // Assign detector IDs for each ("logical" in Geant4) volume
            CELER_EXPECT(geo);
            std::vector<DetectorId> temp_det(geo->num_volumes(), DetectorId{});
            for (auto const& kv : detector_map)
            {
                CELER_ASSERT(kv.first < temp_det.size());
                temp_det[kv.first.unchecked_get()] = kv.second;
            }

            make_builder(&host_data.detector)
                .insert_back(temp_det.begin(), temp_det.end());

            host_data.nonzero_energy_deposition = nonzero_energy_deposition;
        }

        storage_->obj = {std::move(host_data), num_streams};
    }

    if (selection.points[StepPoint::pre] || !detector_map.empty())
    {
        // Some pre-step data is being gathered
        pre_action_
            = std::make_shared<detail::StepGatherAction<StepPoint::pre>>(
                action_registry->next_id(), storage_, VecInterface{});
        action_registry->insert(pre_action_);
    }

    // Always add post-step action, and add callbacks to it
    post_action_ = std::make_shared<detail::StepGatherAction<StepPoint::post>>(
        action_registry->next_id(), storage_, std::move(callbacks));
    action_registry->insert(post_action_);
}

//---------------------------------------------------------------------------//
//!@{
//! Default destructor and move and copy
StepCollector::~StepCollector() = default;
StepCollector::StepCollector(StepCollector const&) = default;
StepCollector& StepCollector::operator=(StepCollector const&) = default;
StepCollector::StepCollector(StepCollector&&) = default;
StepCollector& StepCollector::operator=(StepCollector&&) = default;
//!@}

//---------------------------------------------------------------------------//
/*!
 * See which data are being gathered.
 */
StepSelection const& StepCollector::selection() const
{
    return storage_->obj.params<MemSpace::host>().selection;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
