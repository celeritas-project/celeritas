//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/HeuristicGeoExecutor.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/Atomics.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

#if !CELER_DEVICE_SOURCE
#    include "corecel/cont/ArrayIO.hh"
#endif

#include "HeuristicGeoData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

struct HeuristicGeoExecutor
{
    using ParamsRef = NativeCRef<HeuristicGeoParamsData>;
    using StateRef = NativeRef<HeuristicGeoStateData>;

    ParamsRef params;
    StateRef state;

    CELER_FUNCTION void operator()(ThreadId tid) const
    {
        return (*this)(TrackSlotId{tid.unchecked_get()});
    }
    inline CELER_FUNCTION void operator()(TrackSlotId tid) const;
};

//---------------------------------------------------------------------------//
/*!
 * Take a heuristic step.
 *
 * This is meant to emulate some of the variability in step sizes and direction
 * changes seen during a real transport loop,
 */
CELER_FUNCTION void HeuristicGeoExecutor::operator()(TrackSlotId tid) const
{
    RngEngine rng(params.rng, state.rng, tid);
    GeoTrackView geo(params.geometry, state.geometry, tid);
    if (state.status[tid] == LifeStatus::unborn)
    {
        // Initialize isotropically and uniformly in the box
        UniformBoxDistribution<> sample_pos(params.s.lower, params.s.upper);
        IsotropicDistribution<> sample_dir;

        // Note that pos/dir sampling can't be done as arguments to the same
        // function since the ordering would be unspecified
        GeoTrackView::Initializer_t init;
        init.pos = sample_pos(rng);
        init.dir = sample_dir(rng);
        geo = init;
#if !CELER_DEVICE_SOURCE
        CELER_VALIDATE(!geo.is_outside(),
                       << "failed to initialize at " << init.pos);
#else
        CELER_ASSERT(!geo.is_outside());
#endif

        state.status[tid] = LifeStatus::alive;
    }
    else if (state.status[tid] == LifeStatus::dead)
    {
        return;
    }

    // Sample step length uniformly in log space
    real_type step;
    {
        UniformRealDistribution<> sample_logstep{params.s.log_min_step,
                                                 params.s.log_max_step};
        step = std::exp(sample_logstep(rng));
    }

    // Calculate latest safety and truncate estimated step length (MSC-like)
    // half the time
    if (!geo.is_on_boundary() && geo.volume_id() != params.s.world_volume)
    {
        real_type safety = geo.find_safety();
        CELER_ASSERT(safety >= 0);
        if (safety > params.s.geom_limit)
        {
            BernoulliDistribution truncate_to_safety(0.5);
            if (truncate_to_safety(rng))
            {
                // Safety scaling factor is like "safety_tol" in MSC
                step = min(step, safety * (1 - params.s.safety_tol));
            }
        }
    }

    // Move to boundary and accumulate step
    {
        Propagation prop = geo.find_next_step(step);

        if (prop.boundary)
        {
            geo.move_to_boundary();
            CELER_ASSERT(geo.is_on_boundary());
        }
        else
        {
            // Check for similar assertions in FieldPropagator before loosening
            // this one!
            CELER_ASSERT(prop.distance == step);
            CELER_ASSERT(prop.distance > 0);
#if CELERITAS_DEBUG
            auto orig_pos = geo.pos();
#endif
            geo.move_internal(prop.distance);
            CELER_ASSERT(!geo.is_on_boundary());
#if CELERITAS_DEBUG
            CELER_ASSERT(orig_pos != geo.pos());
#endif
        }

        CELER_ASSERT(geo.volume_id() < state.accum_path.size());
        atomic_add(&state.accum_path[geo.volume_id()], prop.distance);
    }

    BernoulliDistribution do_scatter(0.1);
    if (do_scatter(rng))
    {
        // Forward scatter: anything up to a 90 degree angle if not on a
        // boundary, otherwise pretty close to forward peaked
        real_type min_angle = (geo.is_on_boundary() ? real_type(0.9) : 0);
        real_type mu = UniformRealDistribution<>{min_angle, 1}(rng);
        real_type phi
            = UniformRealDistribution<>{0, real_type(2 * constants::pi)}(rng);

        Real3 dir = geo.dir();
        dir = rotate(from_spherical(mu, phi), dir);
        geo.set_dir(dir);
    }

    if (geo.is_on_boundary())
    {
        geo.cross_boundary();
        CELER_ASSERT(geo.is_on_boundary());

        if (geo.is_outside())
        {
            state.status[tid] = LifeStatus::dead;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
