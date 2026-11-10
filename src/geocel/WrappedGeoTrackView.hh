//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/WrappedGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <type_traits>
#include <utility>

#include "GeoTrackInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a track view for CPU testing and interface validation.
 *
 * \note This uses composition to wrap the parent track view and exposes it
 * through the virtual interface for C++ testing. It should have zero overhead
 * when used directly due to the 'final' class, but it is limited only to
 * running on host.
 *
 * The wrapper stores copies of the (lightweight) params and state references
 * so that the underlying track view can be reconstructed in place by
 * \c set_track_slot , allowing a single heap-allocated instance to be reused
 * for all track slots of a state.
 */
template<class GTV>
class WrappedGeoTrackView final
    : public GeoTrackInterface<typename GTV::real_type>
{
    using Base = GeoTrackInterface<typename GTV::real_type>;

  public:
    //!@{
    //! \name Type aliases
    using GeoTrackViewT = GTV;
    using Initializer_t = GeoTrackInitializer;
    using real_type = typename GTV::real_type;
    using Real3 = typename Base::Real3;
    using VolPathVisitor = typename Base::VolPathVisitor;
    using ParamsRef = typename GTV::ParamsRef;
    using StateRef = typename GTV::StateRef;
    //!@}

  public:
    //! Construct from params and state data and a track slot
    WrappedGeoTrackView(
        ParamsRef const& params, StateRef const& states, TrackSlotId tid)
        : params_{params}, states_{states}, tid_{tid}
    {
        t_.emplace(params_, states_, tid_);
    }

    //! Construct a view to the same track (rebinding to this instance's refs)
    WrappedGeoTrackView(WrappedGeoTrackView&& other)
        : params_{other.params_}, states_{other.states_}, tid_{other.tid_}
    {
        t_.emplace(params_, states_, tid_);
    }

    //! Prohibit assignment and copying
    WrappedGeoTrackView& operator=(WrappedGeoTrackView&&) = delete;
    WrappedGeoTrackView(WrappedGeoTrackView const&) = delete;
    WrappedGeoTrackView& operator=(WrappedGeoTrackView const&) = delete;

    //! Access the underlying track view
    CELER_FORCEINLINE GTV const& track_view() const { return *t_; }
    //! Access the underlying track view
    CELER_FORCEINLINE GTV& track_view() { return *t_; }

    //// GEO TRACK INTERFACE ////

    // Initialize the state
    WrappedGeoTrackView& operator=(Initializer_t const& init) final
    {
        *t_ = init;
        return *this;
    }

    // Rebind the underlying view to another track slot
    void set_track_slot(TrackSlotId tid) final
    {
        t_.emplace(params_, states_, tid);
        tid_ = tid;
    }

    // Physical state
    Real3 const& pos() const final { return t_->pos(); }
    Real3 const& dir() const final { return t_->dir(); }

    // Canonical volume state
    VolumeId volume_id() const final { return t_->volume_id(); }
    VolumeInstanceId volume_instance_id() const final
    {
        return t_->volume_instance_id();
    }
    VolumeLevelId volume_level() const final { return t_->volume_level(); }
    void volume_instance_id(Span<VolumeInstanceId> levels) const final
    {
        t_->volume_instance_id(levels);
    }
    void foreach_volume_path(VolPathVisitor visit) const final
    {
        t_->foreach_volume_path(std::move(visit));
    }

    // Implementation volume ID
    ImplVolumeId impl_volume_id() const final { return t_->impl_volume_id(); }

    // State flags
    bool is_outside() const final { return t_->is_outside(); }
    bool failed() const final { return t_->failed(); }

    // Surface state
    bool is_on_boundary() const final { return t_->is_on_boundary(); }
    Real3 normal() const final { return t_->normal(); }
    GeoStatus geo_status() const final
    {
        if constexpr (HasGeoStatus_<GTV>::value)
        {
            return t_->geo_status();
        }
        else
        {
            return Base::geo_status();
        }
    }

    // Straight-line movement and boundary crossing
    Propagation find_next_step(real_type max_step) final
    {
        return t_->find_next_step(max_step);
    }
    void move_internal(real_type step) final { t_->move_internal(step); }
    void move_to_boundary() final { t_->move_to_boundary(); }
    void cross_boundary() final { t_->cross_boundary(); }

    // Locally bounded movement
    real_type find_safety() final { return t_->find_safety(); }
    real_type find_safety(real_type max_step) final
    {
        return t_->find_safety(max_step);
    }
    void set_dir(Real3 const& newdir) final { t_->set_dir(newdir); }
    void move_internal(Real3 const& pos) final { t_->move_internal(pos); }

  private:
    ParamsRef params_;
    StateRef states_;
    TrackSlotId tid_;
    std::optional<GTV> t_;

    // Determine whether the track has a geo_status accessor
    template<class T, class = void>
    struct HasGeoStatus_ : std::false_type
    {
    };
    template<class T>
    struct HasGeoStatus_<
        T,
        std::void_t<decltype(std::declval<T const&>().geo_status())>>
        : std::true_type
    {
    };
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
