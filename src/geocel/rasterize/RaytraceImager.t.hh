//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/*!
 * \file geocel/rasterize/RaytraceImager.t.hh
 * \brief Template definition file for \c RaytraceImager
 *
 * Include this file in a .cc file and instantiate it explicitly. When
 * instantiating, you must provide access to the GeoTraits specialization as
 * well as the data classes and track view.
 */
//---------------------------------------------------------------------------//
#pragma once

#include "RaytraceImager.hh"

#include "corecel/data/CollectionStateStore.hh"
#include "corecel/sys/MultiExceptionHandler.hh"

#include "Image.hh"

#include "detail/RaytraceExecutor.hh"

#define CELER_INST_RAYTRACE_IMAGER

namespace celeritas
{
//---------------------------------------------------------------------------//
template<class G>
struct RaytraceImager<G>::CachedStates
{
    template<MemSpace M>
    using StateStore = CollectionStateStore<GeoStateData, M>;

    StateStore<MemSpace::host> host;
    StateStore<MemSpace::device> device;

    //! Access the states for the given memspace
    template<MemSpace M>
    StateStore<M>& get()
    {
        if constexpr (M == MemSpace::host)
        {
            return host;
        }
        else
        {
            return device;
        }
    }
};

//---------------------------------------------------------------------------//
/*!
 * Construct with geometry.
 */
template<class G>
RaytraceImager<G>::RaytraceImager(SPGeometry geo)
    : geo_{std::move(geo)}, cache_{std::make_shared<CachedStates>()}
{
    CELER_EXPECT(geo_);
    CELER_ENSURE(cache_);
}

//---------------------------------------------------------------------------//
/*!
 * Raytrace an image on host or device.
 */
template<class G>
void RaytraceImager<G>::operator()(Image<MemSpace::host>* image)
{
    return this->raytrace_impl(image);
}

//---------------------------------------------------------------------------//
/*!
 * Raytrace an image on host or device.
 */
template<class G>
void RaytraceImager<G>::operator()(Image<MemSpace::device>* image)
{
    return this->raytrace_impl(image);
}

//---------------------------------------------------------------------------//
/*!
 * Raytrace an image on host or device.
 */
template<class G>
template<MemSpace M>
void RaytraceImager<G>::raytrace_impl(Image<M>* image)
{
    CELER_EXPECT(image);

    auto const& img_params = *image->params();
    auto const& geo_params = *geo_;
    auto& geo_state_store = cache_->template get<M>();

    if (img_params.num_lines() != geo_state_store.size())
    {
        using StateStore = typename CachedStates::template StateStore<M>;

        // Allocate (or deallocate and reallocate) geometry states
        if (geo_state_store)
        {
            geo_state_store = {};
        }
        geo_state_store
            = StateStore{geo_params.host_ref(), img_params.num_lines()};
    }

    // Raytrace it!
    this->launch_raytrace_kernel(geo_params.template ref<M>(),
                                 geo_state_store.ref(),
                                 img_params.template ref<M>(),
                                 image->ref());
}

//---------------------------------------------------------------------------//
/*!
 * Execute the raytrace on the host.
 */
template<class G>
void RaytraceImager<G>::launch_raytrace_kernel(
    GeoParamsCRef<MemSpace::host> const& geo_params,
    GeoStateRef<MemSpace::host> const& geo_states,
    ImageParamsCRef<MemSpace::host> const& img_params,
    ImageStateRef<MemSpace::host> const& img_state) const
{
    using CalcId = detail::VolumeIdCalculator;
    using Executor = detail::RaytraceExecutor<GeoTrackView, CalcId>;
    Executor execute_thread{
        geo_params, geo_states, img_params, img_state, CalcId{}};

    size_type const num_threads = geo_states.size();

    MultiExceptionHandler capture_exception;
#if CELERITAS_OPENMP == CELERITAS_OPENMP_TRACK
#    pragma omp parallel for
#endif
    for (size_type i = 0; i < num_threads; ++i)
    {
        CELER_TRY_HANDLE(execute_thread(ThreadId{i}), capture_exception);
    }
    log_and_rethrow(std::move(capture_exception));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the raytrace on the device.
 */
#if !CELER_USE_DEVICE
template<class G>
void RaytraceImager<G>::launch_raytrace_kernel(
    GeoParamsCRef<MemSpace::device> const&,
    GeoStateRef<MemSpace::device> const&,
    ImageParamsCRef<MemSpace::device> const&,
    ImageStateRef<MemSpace::device> const&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
