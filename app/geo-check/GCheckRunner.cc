//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file GCheckRunner.cc
//---------------------------------------------------------------------------//
#include "GCheckRunner.hh"

//#include "base/Range.hh"
#include "base/Stopwatch.hh"
#include "base/ColorUtils.hh"
#include "comm/Logger.hh"
#include "geometry/GeoParams.hh"
#include "geometry/GeoStateStore.hh"
#include "geometry/GeoInterface.hh"
#include "geometry/GeoTrackView.hh"
#include "geometry/LinearPropagator.hh"
#include "GCheckKernel.hh"
#include "VecGeom/navigation/NavigationState.h"

using namespace celeritas;

namespace geo_check
{
using NavState = vecgeom::NavigationState;

//---------------------------------------------------------------------------//
/*!
 * Construct with image parameters
 */
GCheckRunner::GCheckRunner(SPConstGeo geometry, int max_steps)
    : geo_params_(std::move(geometry)), max_steps_(max_steps)
{
    CELER_EXPECT(geo_params_);
    CELER_EXPECT(max_steps > 0);
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a track for debugging purposes
 */
void GCheckRunner::operator()(const GeoStateInitializer* init, int ntrks) const
{
    CELER_EXPECT(init);

    GeoStatePointers state_view;
    state_view.size       = ntrks;
    state_view.vgmaxdepth = geo_params_->max_depth();
    state_view.pos        = new Real3;
    state_view.dir        = new Real3;
    state_view.next_step  = new real_type{0.0};
    state_view.vgstate    = NavState::MakeInstance(state_view.vgmaxdepth);
    state_view.vgnext     = NavState::MakeInstance(state_view.vgmaxdepth);

    // run on the CPU
    CELER_LOG(status) << "Propagating track(s) on CPU";
    run_cpu(geo_params_->host_pointers(), state_view, init, max_steps_);

    // run on the GPU device
    GeoStateStore geo_state(*geo_params_, ntrks);

    CELER_LOG(status) << "Propagating track(s) on GPU";
    run_gpu(geo_params_->device_pointers(),
            geo_state.device_pointers(),
            *init,
            max_steps_);
}

//---------------------------------------------------------------------------//
} // namespace geo_check
