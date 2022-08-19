//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/geo/Geometry.test.hh
//---------------------------------------------------------------------------//

#include <cmath>

#include "celeritas_config.h"
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/Atomics.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/random/RngData.hh"
#include "celeritas/random/RngEngine.hh"
#include "celeritas/random/distribution/BernoulliDistribution.hh"
#include "celeritas/random/distribution/IsotropicDistribution.hh"
#include "celeritas/random/distribution/ReciprocalDistribution.hh"
#include "celeritas/random/distribution/UniformBoxDistribution.hh"
#include "celeritas/random/distribution/UniformRealDistribution.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// DATA
//---------------------------------------------------------------------------//
struct GeoTestScalars
{
    // User-configurable options
    Real3     lower{0, 0, 0};
    Real3     upper{0, 0, 0};
    real_type log_min_step{-16.11809565095832}; // 1 nm
    real_type log_max_step{2.302585092994046};  // 10 cm

    // Set from geometry
    VolumeId::size_type num_volumes{};
    bool                ignore_zero_safety = !CELERITAS_USE_VECGEOM;

    explicit CELER_FUNCTION operator bool() const
    {
        return log_min_step <= log_max_step && num_volumes > 0;
    }
};

template<Ownership W, MemSpace M>
struct GeoTestParamsData
{
    GeoParamsData<W, M> geometry;
    RngParamsData<W, M> rng;
    GeoTestScalars      s;

    //! True if all params are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && rng && s;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeoTestParamsData& operator=(const GeoTestParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry = other.geometry;
        rng      = other.rng;
        s        = other.s;
        return *this;
    }
};

// Special enum to avoid std::vector<bool>
enum LifeStatus : bool
{
    dead = 0,
    alive,
};

template<Ownership W, MemSpace M>
struct GeoTestStateData
{
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;

    GeoStateData<W, M>     geometry;
    RngStateData<W, M>     rng;
    StateItems<LifeStatus> alive;

    celeritas::Collection<real_type, W, M, VolumeId> accum_path;

    //! Number of state elements
    CELER_FUNCTION size_type size() const { return geometry.size(); }

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return geometry && rng && !alive.empty() && !accum_path.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    GeoTestStateData& operator=(GeoTestStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        geometry   = other.geometry;
        rng        = other.rng;
        alive      = other.alive;
        accum_path = other.accum_path;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize states in host code.
 */
template<MemSpace M>
inline void resize(GeoTestStateData<Ownership::value, M>* state,
                   const HostCRef<GeoTestParamsData>&     params,
                   size_type                              size)
{
    CELER_EXPECT(state);
    CELER_EXPECT(params);
    CELER_EXPECT(size > 0);
    resize(&state->geometry, params.geometry, size);
    resize(&state->rng, params.rng, size);
    resize(&state->alive, size);
    fill(LifeStatus::dead, &state->alive);

    resize(&state->accum_path, params.s.num_volumes);
    fill(real_type{0}, &state->accum_path);
}

//---------------------------------------------------------------------------//

struct GeoTestLauncher
{
    using ParamsRef = NativeCRef<GeoTestParamsData>;
    using StateRef  = NativeRef<GeoTestStateData>;

    const ParamsRef& params;
    const StateRef&  state;

    inline CELER_FUNCTION void operator()(ThreadId tid) const;
};

//---------------------------------------------------------------------------//
// LAUNCHER
//---------------------------------------------------------------------------//
/*!
 * Take a heuristic step.
 *
 * This is meant to emulate some of the variability in step sizes and direction
 * changes seen during a real transport loop,
 */
CELER_FUNCTION void GeoTestLauncher::operator()(ThreadId tid) const
{
    RngEngine    rng(state.rng, tid);
    GeoTrackView geo(params.geometry, state.geometry, tid);
    if (state.alive[tid] == LifeStatus::dead)
    {
        // Initialize isotropically and uniformly in the box
        UniformBoxDistribution<> sample_pos(params.s.lower, params.s.upper);
        IsotropicDistribution<>  sample_dir;

        // Note that pos/dir sampling can't be done as arguments to the same
        // function since the ordering would be unspecified
        GeoTrackView::Initializer_t init;
        init.pos = sample_pos(rng);
        init.dir = sample_dir(rng);
        geo      = init;

        state.alive[tid] = LifeStatus::alive;
    }

    // Sample step length uniformly in log space
    real_type step;
    {
        UniformRealDistribution<> sample_logstep{params.s.log_min_step,
                                                 params.s.log_max_step};
        step = std::exp(sample_logstep(rng));
    }

    // Calculate safety and truncate estimated step length
    {
        real_type safety = geo.find_safety();
        CELER_ASSERT(safety >= 0);
        if (params.s.ignore_zero_safety && safety == 0)
        {
            safety = numeric_limits<real_type>::infinity();
        }

        BernoulliDistribution truncate_to_safety(0.5);
        if (truncate_to_safety(rng))
        {
            step = min(step, safety * real_type(1.5) + real_type(1e-9));
        }
    }

    // Find step length
    Propagation prop = geo.find_next_step(step);

    if (prop.boundary)
    {
        geo.move_to_boundary();
        CELER_ASSERT(geo.is_on_surface());
    }
    else
    {
        // Check for similar assertions in FieldPropagator before loosening
        // this one!
        CELER_ASSERT(prop.distance == step);
        geo.move_internal(prop.distance);
        CELER_ASSERT(!geo.is_on_surface());
    }

    BernoulliDistribution do_scatter(0.1);
    if (do_scatter(rng))
    {
        // Forward scatter: anything up to a 90 degree angle if not on a
        // boundary, otherwise pretty close to forward peaked
        real_type min_angle = (prop.boundary ? real_type(0.9) : 0);
        real_type mu        = UniformRealDistribution<>{min_angle, 1}(rng);
        real_type phi
            = UniformRealDistribution<real_type>{0, 2 * constants::pi}(rng);

        Real3 dir = geo.dir();
        rotate(from_spherical(mu, phi), dir);
        geo.set_dir(dir);
    }

    if (prop.boundary)
    {
        geo.cross_boundary();
        CELER_ASSERT(geo.is_on_surface());
    }

    if (geo.is_outside())
    {
        state.alive[tid] = LifeStatus::dead;
    }
    else
    {
        CELER_ASSERT(geo.volume_id() < state.accum_path.size());
        atomic_add(&state.accum_path[geo.volume_id()], prop.distance);
    }
}
//---------------------------------------------------------------------------//
// DEVICE KERNEL EXECUTION
//---------------------------------------------------------------------------//
//! Run on device
void g_test(const DeviceCRef<GeoTestParamsData>&,
            const DeviceRef<GeoTestStateData>&);

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline void g_test(const DeviceCRef<GeoTestParamsData>&,
                   const DeviceRef<GeoTestStateData>&)
{
    {
        CELER_NOT_CONFIGURED("CUDA or HIP");
    }
}
#endif

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
