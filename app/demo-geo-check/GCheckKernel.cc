//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file demo-geo-check/GCheckKernel.cc
//---------------------------------------------------------------------------//
#include "GCheckKernel.hh"

#include <cstdio>
#include <type_traits>

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/sys/ThreadId.hh"
#include "orange/OrangeData.hh"
#include "celeritas/field/LinearPropagator.hh"
#include "celeritas/geo/GeoParams.hh"  // IWYU pragma: keep

using std::printf;

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 *  Run tracking on the CPU
 */
GCheckOutput run_cpu(SPConstGeo const& params,
                     GeoTrackInitializer const* init,
                     int max_steps)
{
    using StateStore = CollectionStateStore<GeoStateData, MemSpace::host>;

    StateStore state = StateStore(params->host_ref(), 1);

    GeoTrackView geo(params->host_ref(), state.ref(), TrackSlotId(0));
    geo = GeoTrackInitializer{init->pos, init->dir};

    LinearPropagator propagate(geo);  // one propagator per track

    printf("Initial track: pos=(%f, %f, %f), dir=(%f, %f, %f), outside=%i\n",
           geo.pos()[0],
           geo.pos()[1],
           geo.pos()[2],
           geo.dir()[0],
           geo.dir()[1],
           geo.dir()[2],
           geo.is_outside());

    // Track from outside detector, moving right
    GCheckOutput result;
    int istep = 0;
    int stuck = 0;
    do
    {
        auto step = propagate();  // to next boundary
        if (step.boundary)
            geo.cross_boundary();
        result.ids.push_back(physid(geo));
        result.distances.push_back(step.distance);

        // tag stuck tracks
        if (step.distance < 1.0e-6)
        {
            ++stuck;
            if (stuck > 8)
            {
                printf(
                    "stuck @step=%i/%i: pos=(%g; %g; %g) - step=%g nxStep=%g "
                    "volID=%i(%i)\n",
                    stuck,
                    istep,
                    geo.pos()[0],
                    geo.pos()[1],
                    geo.pos()[2],
                    step.distance,
                    geo.find_next_step().distance,
                    physid(geo),
                    static_cast<int>(geo.volume_id().get()));
            }
        }
        else
        {
            stuck = 0;
        }
        ++istep;
    } while (!geo.is_outside() && istep < max_steps);

    auto& temp = geo.pos();
    printf("End of loop after %i steps @ pos=(%g; %g; %g)\n",
           istep,
           temp[0],
           temp[1],
           temp[2]);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
