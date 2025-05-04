//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepParams.cc
//---------------------------------------------------------------------------//
#include "StepParams.hh"

#include <vector>

#include "corecel/data/AuxStateData.hh"
#include "corecel/data/AuxStateVec.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Label.hh"
#include "celeritas/geo/GeoParams.hh"

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
                       GeoParams const& geo,
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

    auto const& volumes = geo.volumes();
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
            CELER_VALIDATE(inserted,
                           << "multiple step interfaces map single volume "
                              "to a detector ('"
                           << volumes.at(prev->first) << "' -> "
                           << prev->second.get() << " and '"
                           << volumes.at(kv.first) << "' -> "
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

    // Save data
    mirror_ = CollectionMirror{[&] {
        HostVal<StepParamsData> host_data;
        host_data.selection = selection;

        if (!detector_map.empty())
        {
            // Assign detector IDs for each ("logical" in Geant4) volume
            std::vector<DetectorId> temp_det(geo.volumes().size(),
                                             DetectorId{});
            for (auto const& kv : detector_map)
            {
                CELER_ASSERT(kv.first < temp_det.size());
                temp_det[kv.first.unchecked_get()] = kv.second;
            }

            CollectionBuilder{&host_data.detector}.insert_back(
                temp_det.begin(), temp_det.end());

            host_data.nonzero_energy_deposition = nonzero_energy_deposition;
        }

        if (selection.points[StepPoint::pre].volume_instance_ids
            || selection.points[StepPoint::post].volume_instance_ids)
        {
            host_data.volume_instance_depth = geo.max_depth();
            CELER_VALIDATE(host_data.volume_instance_depth > 0,
                           << "geometry type does not support volume "
                              "instance IDs: max depth is "
                           << host_data.volume_instance_depth);
        }

        return host_data;
    }()};

    CELER_ASSERT((has_det == HasDetectors::all) == this->has_detectors());
}

//---------------------------------------------------------------------------//
/*!
 * Build state data for a stream.
 */
auto StepParams::create_state(MemSpace m,
                              StreamId stream,
                              size_type size) const -> UPState
{
    return make_aux_state<StepStateData>(*this, m, stream, size);
}

//---------------------------------------------------------------------------//
/*!
 * Access the step state from a state vector.
 */
template<MemSpace M>
StepStateData<Ownership::reference, M>&
StepParams::state_ref(AuxStateVec& aux) const
{
    using StateT = AuxStateData<StepStateData, M>;
    return get<StateT>(aux, aux_id_).ref();
}

//---------------------------------------------------------------------------//
// EXPLICIT TEMPLATE INSTANTIATION
//---------------------------------------------------------------------------//

template HostRef<StepStateData>& StepParams::state_ref(AuxStateVec&) const;

template DeviceRef<StepStateData>& StepParams::state_ref(AuxStateVec&) const;

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
