//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/CoreTrackView.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/MieData.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/interactor/MieInteractor.hh"
#include <cstdio>
#include "celeritas/optical/detail/OpticalKillTally.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
struct MieExecutor
{
    inline CELER_FUNCTION Interaction operator()(CoreTrackView const&);

    NativeCRef<MieData> data;
};

//---------------------------------------------------------------------------//
/*!
 * Sample optical Mie interaction from the current track.
 */
CELER_FUNCTION Interaction MieExecutor::operator()(CoreTrackView const& track)
{
    CELER_EXPECT(data);

    // Access the current particle track (optical photon)
    auto particle = track.particle();
    // Photon's current direction
    auto const& direction = track.geometry().dir();
    // RNG stream for sampling scattering
    auto rng = track.rng();
    // Look up the Mie parameters for this material
    auto matid = track.material_record().material_id();
    CELER_ASSERT(matid < data.mie_record.size());

#if !CELER_DEVICE_COMPILE
    celeritas::optical::detail::tally_optical_kill(
        "mie-scatter",
        track.geometry().volume_id().unchecked_get(),
        track.particle().energy().value() > 4.576e-6);
    if (celeritas::optical::detail::surface_trace_enabled()
        && static_cast<int>(track.track_slot_id().get())
               == celeritas::optical::detail::traced_slot().load())
    {
        char buf[192];
        std::snprintf(buf,
                      sizeof(buf),
                      "slot%d MIE vol=%u xyz=(%.5f,%.5f,%.5f) dir=(%.3f,%.3f,%.3f)",
                      static_cast<int>(track.track_slot_id().get()),
                      track.geometry().volume_id().unchecked_get(),
                      track.geometry().pos()[0],
                      track.geometry().pos()[1],
                      track.geometry().pos()[2],
                      track.geometry().dir()[0],
                      track.geometry().dir()[1],
                      track.geometry().dir()[2]);
        celeritas::optical::detail::trace_surface(buf);
    }
#endif

    MieInteractor interact{data, particle, direction, matid};

    return interact(rng);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
