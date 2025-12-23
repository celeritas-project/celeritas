//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/WrappedGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "geocel/GeoTrackInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a track view for CPU testing and interface validation.
 *
 * \note This uses composition to wrap the parent track view and exposes it
 * through the virtual interface for C++ testing.
 */
template<class GTV>
class WrappedGeoTrackView final
    : public GeoTrackInterface<typename GTV::real_type>
{
  public:
    //!@{
    //! \name Type aliases
    using GeoTrackViewT = GTV;
    using Initializer_t = GeoTrackInitializer;
    using real_type = typename GTV::real_type;
    using Real3 = typename GeoTrackInterface<typename GTV::real_type>::Real3;
    //!@}

  public:
    //! Forward construction arguments to the original track view
    template<class... Args>
    WrappedGeoTrackView(Args&&... args) : t_(std::forward<Args>(args)...)
    {
    }

    CELER_DEFAULT_MOVE_DELETE_COPY(WrappedGeoTrackView);

    //! Access the underlying track view
    CELER_FORCEINLINE GTV const& track_view() const { return t_; }
    //! Access the underlying track view
    CELER_FORCEINLINE GTV& track_view() { return t_; }

    //// GEO TRACK INTERFACE ////

    // Initialize the state
    WrappedGeoTrackView& operator=(Initializer_t const& init) final
    {
        t_ = init;
        return *this;
    }

    // Physical state
    Real3 const& pos() const final { return t_.pos(); }
    Real3 const& dir() const final { return t_.dir(); }

    // Canonical volume state
    VolumeId volume_id() const final { return t_.volume_id(); }
    VolumeInstanceId volume_instance_id() const final
    {
        return t_.volume_instance_id();
    }
    VolumeLevelId volume_level() const final { return t_.volume_level(); }
    void volume_instance_id(Span<VolumeInstanceId> levels) const final
    {
        t_.volume_instance_id(levels);
    }

    // Implementation volume ID
    ImplVolumeId impl_volume_id() const final { return t_.impl_volume_id(); }

    // State flags
    bool is_outside() const final { return t_.is_outside(); }
    bool failed() const final { return t_.failed(); }

    // Surface state
    bool is_on_boundary() const final { return t_.is_on_boundary(); }
    Real3 normal() const final { return t_.normal(); }

    // Straight-line movement and boundary crossing
    Propagation find_next_step() final { return t_.find_next_step(); }
    Propagation find_next_step(real_type max_step) final
    {
        return t_.find_next_step(max_step);
    }
    void move_internal(real_type step) final { t_.move_internal(step); }
    void move_to_boundary() final { t_.move_to_boundary(); }
    void cross_boundary() final { t_.cross_boundary(); }

    // Locally bounded movement
    real_type find_safety() final { return t_.find_safety(); }
    real_type find_safety(real_type max_step) final
    {
        return t_.find_safety(max_step);
    }
    void set_dir(Real3 const& newdir) final { t_.set_dir(newdir); }
    void move_internal(Real3 const& pos) final { t_.move_internal(pos); }

  private:
    GTV t_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//

template<class GTV>
WrappedGeoTrackView(GTV&&) -> WrappedGeoTrackView<GTV>;

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
