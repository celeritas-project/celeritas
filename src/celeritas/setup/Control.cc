//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/setup/Control.cc
//---------------------------------------------------------------------------//
#include "Control.hh"

#include "corecel/sys/Device.hh"
#include "corecel/sys/MpiCommunicator.hh"
#include "celeritas/global/CoreSizes.hh"
#include "celeritas/inp/Control.hh"
#include "celeritas/optical/OpticalSizes.hh"

namespace celeritas
{
namespace setup
{
//---------------------------------------------------------------------------//
/*!
 * Resolve core state capacity values.
 *
 * This validates the user input and sets default values which may depend on
 * whether the device is enabled.
 */
CoreSizes capacity(inp::CoreStateCapacity const& c, size_type num_streams)
{
    using Defaults = inp::CoreStateCapacity;

    CELER_VALIDATE(!c.tracks || *c.tracks > 0,
                   << "nonpositive capacity.tracks=" << *c.tracks);
    CELER_VALIDATE(!c.primaries || *c.primaries > 0,
                   << "nonpositive capacity.primaries=" << *c.primaries);
    CELER_VALIDATE(!c.initializers || *c.initializers > 0,
                   << "nonpositive capacity.initializers=" << *c.initializers);
    CELER_VALIDATE(!c.secondaries || *c.secondaries > 0,
                   << "nonpositive capacity.secondaries=" << *c.secondaries);
    CELER_VALIDATE(!c.events || *c.events > 0,
                   << "nonpositive capacity.events=" << *c.events);
    CELER_VALIDATE(num_streams > 0,
                   << "control.num_streams must be manually set before setup");

    CoreSizes result;

    if (celeritas::device())
    {
        result.tracks = c.tracks.value_or(Defaults::gpu_tracks);
    }
    else
    {
        result.tracks = c.tracks.value_or(Defaults::cpu_tracks);
    }

    result.primaries
        = c.primaries.value_or(Defaults::primaries_per_track * result.tracks);
    result.initializers = c.initializers.value_or(
        Defaults::initializers_per_track * result.tracks);
    result.secondaries = c.secondaries.value_or(Defaults::secondaries_per_track
                                                * result.tracks);
    result.events = c.events.value_or(1);
    result.streams = num_streams;
    result.processes = comm_world().size();

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Resolve optical state capacity values.
 *
 * This validates the user input and sets default values which may depend on
 * whether the device is enabled.
 */
OpticalSizes capacity(inp::OpticalStateCapacity const& c, size_type num_streams)
{
    using Defaults = inp::OpticalStateCapacity;

    CELER_VALIDATE(!c.tracks || *c.tracks > 0,
                   << "nonpositive capacity.tracks=" << *c.tracks);
    CELER_VALIDATE(!c.primaries || *c.primaries > 0,
                   << "nonpositive capacity.primaries=" << *c.primaries);
    CELER_VALIDATE(!c.generators || *c.generators > 0,
                   << "nonpositive capacity.generators=" << *c.generators);
    CELER_VALIDATE(num_streams > 0,
                   << "p.num_streams must be manually set before setup");

    OpticalSizes result;

    if (celeritas::device())
    {
        result.tracks = c.tracks.value_or(Defaults::gpu_tracks);
    }
    else
    {
        result.tracks = c.tracks.value_or(Defaults::cpu_tracks);
    }

    result.primaries
        = c.primaries.value_or(Defaults::primaries_per_track * result.tracks);
    result.generators = c.generators.value_or(Defaults::generators_per_track
                                              * result.tracks);
    result.streams = num_streams;

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace setup
}  // namespace celeritas
