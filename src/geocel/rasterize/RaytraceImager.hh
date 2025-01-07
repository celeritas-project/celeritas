//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/rasterize/RaytraceImager.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "geocel/GeoTraits.hh"

#include "ImageInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
template<Ownership, MemSpace>
struct ImageParamsData;
template<Ownership, MemSpace>
struct ImageStateData;

//---------------------------------------------------------------------------//
/*!
 * Generate one or more images from a geometry by raytracing.
 */
template<class G>
class RaytraceImager final : public ImagerInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPGeometry = std::shared_ptr<G const>;
    //!@}

  public:
    // Construct with geometry
    explicit RaytraceImager(SPGeometry geo);
    ~RaytraceImager() final = default;

    // Raytrace an image on host or device
    void operator()(Image<MemSpace::host>* image) final;
    void operator()(Image<MemSpace::device>* image) final;

  private:
    //// TYPES ////

    using GTraits = GeoTraits<G>;
    template<Ownership W, MemSpace M>
    using GeoStateData = typename GTraits::template StateData<W, M>;
    template<Ownership W, MemSpace M>
    using GeoParamsData = typename GTraits::template ParamsData<W, M>;
    using GeoTrackView = typename GTraits::TrackView;

    template<MemSpace M>
    using GeoParamsCRef = GeoParamsData<Ownership::const_reference, M>;
    template<MemSpace M>
    using GeoStateRef = GeoStateData<Ownership::reference, M>;
    template<MemSpace M>
    using ImageParamsCRef = ImageParamsData<Ownership::const_reference, M>;
    template<MemSpace M>
    using ImageStateRef = ImageStateData<Ownership::reference, M>;

    struct CachedStates;

    //// DATA ////

    SPGeometry geo_;
    std::shared_ptr<CachedStates> cache_;

    //// MEMBER FUNCTIONS ////

    CELER_DEFAULT_MOVE_DELETE_COPY(RaytraceImager);

    template<MemSpace M>
    void raytrace_impl(Image<M>* image);

    void
    launch_raytrace_kernel(GeoParamsCRef<MemSpace::host> const& geo_params,
                           GeoStateRef<MemSpace::host> const& geo_states,
                           ImageParamsCRef<MemSpace::host> const& img_params,
                           ImageStateRef<MemSpace::host> const& img_state) const;

    void launch_raytrace_kernel(
        GeoParamsCRef<MemSpace::device> const& geo_params,
        GeoStateRef<MemSpace::device> const& geo_states,
        ImageParamsCRef<MemSpace::device> const& img_params,
        ImageStateRef<MemSpace::device> const& img_state) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
