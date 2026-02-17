//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/CheckedGeoTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <optional>

#include "geocel/GeoTrackInterface.hh"
#include "geocel/Types.hh"

#include "UnitUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class GeoParamsInterface;
class VolumeParams;

namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Check validity of safety and volume crossings while navigating on CPU.
 *
 * This wraps a \c GeoTrackInterface and adds validation and instrumentation.
 * It counts the number of calls to \c find_next_step and \c find_safety .
 *
 * Two flags can alter the error checking:
 * - \c check_normal will validate the normal calculation when on a boundary
 * - \c check_failure will throw before and after a call
 *
 * The only nontrivial function that can be called from an outside or failed
 * state is initialization (with \c operator= ).
 */
class CheckedGeoTrackView final : public GeoTrackInterface<real_type>
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = GeoTrackInitializer;
    using Real3 = Array<real_type, 3>;
    using TrackT = GeoTrackInterface<real_type>;
    using UPTrack = std::unique_ptr<TrackT>;
    using SPConstGeoI = std::shared_ptr<GeoParamsInterface const>;
    using SPConstVolumes = std::shared_ptr<VolumeParams const>;
    //!@}

  public:
    //! Construct with a unique pointer to a geo track view
    explicit CheckedGeoTrackView(UPTrack track)
        : CheckedGeoTrackView{std::move(track), nullptr, nullptr, UnitLength{}}
    {
    }

    // Construct with a unique pointer to a geo track view
    CheckedGeoTrackView(UPTrack track,
                        SPConstVolumes volumes,
                        SPConstGeoI geo_interface,
                        UnitLength unit_length);

    // Allow moving for customization
    CELER_DEFAULT_MOVE_DELETE_COPY(CheckedGeoTrackView);

    //! Check if we were "moved from" (permanently invalid state)
    explicit operator bool() const { return t_ != nullptr; }

    //! Access the underlying track view
    TrackT const& track_view() const { return *t_; }
    //! Access the underlying track view
    TrackT& track_view() { return *t_; }

    //! Check volume consistency this far from the boundary
    real_type safety_tol() const { return 1e-4 * unit_length_.value; }

    //// ACCESSORS ////

    //! Number of calls of find_next_step
    size_type intersect_count() const { return count_.intersect; }
    //! Number of calls of find_safety
    size_type safety_count() const { return count_.safety; }
    //! Reset the counters
    void reset_count() { count_ = {}; }

    //! Enable/disable normal checking
    void check_normal(bool value) { check_normal_ = value; }
    //! Whether normal checking is enabled
    bool check_normal() const { return check_normal_; }

    //! Enable/disable failure state checking
    void check_failure(bool value) { check_failure_ = value; }
    //! Whether failure checking is enabled
    bool check_failure() const { return check_failure_; }

    //! Enable/disable zero-step distance checking
    void check_zero_distance(bool value) { check_zero_distance_ = value; }
    //! Whether zero-distance checking is enabled
    bool check_zero_distance() const { return check_zero_distance_; }

    //! Canonical volume parameters (if available)
    SPConstVolumes const& volumes() const { return volumes_; }
    //! Geometry interface (if available)
    SPConstGeoI const& geo_interface() const { return geo_interface_; }
    //! Length scale
    UnitLength unit_length() const { return unit_length_; }

    //// GEO TRACK INTERFACE ////

    // Initialize the state
    CheckedGeoTrackView& operator=(GeoTrackInitializer const& init) final;

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

    // Implementation volume ID
    ImplVolumeId impl_volume_id() const final { return t_->impl_volume_id(); }

    // State flags
    bool is_outside() const final { return t_->is_outside(); }
    bool failed() const final { return t_->failed(); }

    // Surface state
    bool is_on_boundary() const final { return t_->is_on_boundary(); }
    Real3 normal() const final { return t_->normal(); }

    // Calculate or return the safety up to an infinite distance
    real_type find_safety() final;

    // Calculate or return the safety up to the given distance
    real_type find_safety(real_type max_safety) final;

    // Change the direction
    void set_dir(Real3 const&) final;

    // Find the distance to the next boundary
    Propagation find_next_step() final;

    // Find the distance to the next boundary
    Propagation find_next_step(real_type max_distance) final;

    // Move a linear step fraction
    void move_internal(real_type) final;

    // Move within the safety distance to a specific point
    void move_internal(Real3 const& pos) final;

    // Move to the boundary in preparation for crossing it
    void move_to_boundary() final;

    // Cross from one side of the current surface to the other
    void cross_boundary() final;

  private:
    UPTrack t_;

    // Metadata
    SPConstVolumes volumes_;
    SPConstGeoI geo_interface_;
    UnitLength unit_length_;

    // Configuration flags
    bool check_normal_{true};
    bool check_failure_{true};
    bool check_safety_{true};
    bool check_zero_distance_{true};

    // Counters
    struct
    {
        size_type intersect{0};
        size_type safety{0};
    } count_;

    // Temporary state
    bool checked_internal_{false};
    std::optional<real_type> next_boundary_;
};

class CheckedGeoError : public RuntimeError
{
  public:
    using RuntimeError::RuntimeError;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Get the descriptive, robust volume name based on the geo state
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        VolumeParams const& params);

// Get a robust name using impl volume params
std::string volume_name(GeoTrackInterface<real_type> const& geo,
                        GeoParamsInterface const& params);

// Get the descriptive, robust volume instance name based on the geo state
std::string volume_instance_name(GeoTrackInterface<real_type> const& geo,
                                 VolumeParams const& params);

// Get the descriptive, robust volume instance name based on the geo state
std::string unique_volume_name(GeoTrackInterface<real_type> const& geo,
                               VolumeParams const& params);

std::ostream& operator<<(std::ostream&, CheckedGeoTrackView const&);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
