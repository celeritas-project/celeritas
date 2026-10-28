//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/WavelengthShiftExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/data/ObserverPtr.hh"
#include "corecel/math/Atomics.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"
#include "celeritas/optical/interactor/WavelengthShiftInteractor.hh"
#include "celeritas/track/CoreStateCounters.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct WavelengthShiftExecutor
{
    inline CELER_FUNCTION Interaction operator()(CoreTrackView const&);

    NativeCRef<WavelengthShiftData> data;
    NativeRef<WlsGeneratorStateData> aux_data;
    size_type buffer_size{};
    ObserverPtr<CoreStateCounters, MemSpace::native> core_counters;
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical WLS interaction from the current track.
 */
CELER_FUNCTION Interaction WavelengthShiftExecutor::operator()(
    CoreTrackView const& track)
{
    auto particle = track.particle();
    auto sim = track.sim();
    auto mat_id = track.material_record().material_id();
    CELER_EXPECT(core_counters);

    auto dist_id = id_cast<ItemId<WlsDistributionData>>(
        buffer_size + track.track_slot_id().get());
    auto rng = track.rng();

    WavelengthShiftInteractor interact{
        data, aux_data, particle, sim, track.geometry().pos(), mat_id, dist_id};
    Interaction result = interact(rng);

    // Count the new photons as pending so the transport loop continues until
    // they are generated
    if (auto const& dist = aux_data.distributions[dist_id])
    {
        atomic_add(&core_counters->num_pending, dist.num_photons);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
