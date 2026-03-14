//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/GeoTrackInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Config.hh"

#include "corecel/Macros.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/NumericLimits.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Standard interface to geometry navigation for a track for testing on CPU.
 * \tparam RealType Floating point precision
 *
 * \note This class is for illustrative and testing purposes \b only
 *   (see celeritas::test::WrappedGeoTrackView) and is
 *   \b not used during the main Celeritas execution. The geometry there is
 *   determined by the \c CELERITAS_CORE_GEO configuration variable and defined
 *   as a type alias \c celeritas::CoreGeoTrackView .
 *
 * Initialization is performed via the assignment operator using a \c
 * GeoTrackInitializer.
 *
 * - Initialization may fail, leaving the track with an "error" status.
 * - Some implementations allow initialization on a boundary, leaving the track
 *   in an "outgoing from boundary" state (i.e., the next step is into the
 *   volume the track initializes inside).
 * - Some implementations allow initialization to succeed but be in an
 *   unphysical part of the geometry (e.g., ORANGE exterior volume). In this
 *   case, the canonical volume ID will be null, indicating no
 *   corresponding user-defined geometry region.
 *
 * Tracking to and across volumes along a straight line requires a specific
 * sequence of calls.
 *
 * - Locate the boundary crossing along the current direction with \c
 *   find_next_step.
 * - Move within the current volume, not crossing a boundary, via \c
 *   move_internal or \c move_to_boundary.
 * - If on a boundary, \c normal can be used to calculate the current surface
 *   normal, but its dot product with the track direction may not be
 *   meaningful. Use \c geo_status to determine whether the track is incident
 *   to or outgoing from the boundary.
 * - If incident into the boundary, change volumes ("relocate") with \c
 *   cross_boundary.  The post-crossing status can be \c error, \c
 *   boundary_out, or \c invalid. Some geometries represent the exterior as a
 *   null canonical VolumeId, and some as an invalid state.
 *
 * \note The free function \c is_on_boundary will be true of the geo status
 * both before \em and after the call to \c cross_boundary, and the surface
 * normal can be calculated in both cases.
 *
 * Movement to a nearby but arbitrary point can be done inside a
 * "safety" distance:
 *
 * - Locate the closest point on the boundary in any direction with \c
 *   find_safety.
 * - Change the direction with \c set_dir. (Note that this will always
 *   invalidate the linear "next step".)
 * - Move to a point with \c move_internal.
 *
 * Neither accessors nor mutators should be called in an \c invalid state.
 * It is allowable but potentially dangerous to call accessors in an \c error
 * state.
 */
template<class RealType = ::celeritas::real_type>
class GeoTrackInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Initializer_t = GeoTrackInitializer;
    using real_type = RealType;
    using Real3 = Array<real_type, 3>;
    //!@}

  public:
    // Anchor virtual destructor in GeoInterface.cc
    virtual ~GeoTrackInterface() = 0;

    /*!
     * Initialize the state.
     *
     * Takes a \c GeoTrackInitializer object to locate the point in the
     * geometry hierarchy.
     *
     * \post \c geo_status() is never \c GeoStatus::boundary_inc : the
     *   result is \c interior (placed inside a volume), \c boundary_out
     *   (placed on a boundary with direction heading away from it), or \c
     *   error (volume not found).
     */
    virtual GeoTrackInterface& operator=(Initializer_t const& init) = 0;

    //!@{
    //! \name Physical state

    //! Return the physical position in the global coordinate system
    virtual Real3 const& pos() const = 0;
    //! Return the direction in the global coordinate system
    virtual Real3 const& dir() const = 0;

    //!@}
    //!@{
    //! \name Canonical volume state
    //! \pre The geo status cannot be \c GeoStatus::invalid

    //! Get the canonical volume ID in the current impl volume
    virtual VolumeId volume_id() const = 0;
    //! Get the physical volume ID in the current cell
    virtual VolumeInstanceId volume_instance_id() const = 0;
    //! Get the distance from root volume in the geometry hierarchy
    virtual VolumeLevelId volume_level() const = 0;
    //! Get the volume instance ID for all levels
    virtual void volume_instance_id(Span<VolumeInstanceId> levels) const = 0;

    /*!
     * Whether the track is outside the valid geometry region.
     *
     * Returns true if the track has left the world (or started outside the
     * outermost known volume).
     *
     * \deprecated Check \c geo.geo_status() is not \c invalid and evaluate \c
     * is_outside(geo.volume_id())
     */
    virtual bool is_outside() const = 0;
    //!@}

    //! Get the implementation volume ID
    virtual ImplVolumeId impl_volume_id() const = 0;

    /*!
     * Whether the last operation resulted in an error.
     *
     * \deprecated check \c geo_status for \c GeoStatus::error instead.
     */
    virtual bool failed() const = 0;

    //!@{
    //! \name Surface state

    /*!
     * Whether the track is exactly on a surface.
     *
     * Returns true if a track is exactly on the boundary of a volume, capable
     * of changing to another volume without altering the physical position.
     *
     * \deprecated use \c is_on_boundary(geo.geo_status())
     */
    virtual bool is_on_boundary() const = 0;

    /*!
     * Calculate the normal vector on the current surface.
     *
     * Returns a global-coordinate direction perpendicular to the volume at
     * the local point when on a boundary. The sign of the surface normal is
     * implementation-dependent; it may change based on the track state
     * (previous volume, direction, surface sign) or geometry construction.
     *
     * \pre \c is_on_boundary(geo.geo_status()) must be true
     */
    virtual Real3 normal() const = 0;

    /*!
     * Derive the geometry state from the existing state flags.
     *
     * This is a shim for geometry implementations that do not natively track
     * the full \c GeoStatus. When on a boundary, it \b assumes the normal
     * vector is oriented "outward" from the current volume.
     * The dot product of the track direction and the surface normal determines
     * whether the track is incident to the boundary (positive: headed outward
     * from the volume) or incident (negative: headed into the volume).
     */
    virtual GeoStatus geo_status() const
    {
        if (this->failed())
            return GeoStatus::error;
        if (this->is_on_boundary())
        {
            return dot_product(this->dir(), this->normal()) >= 0
                       ? GeoStatus::boundary_inc
                       : GeoStatus::boundary_out;
        }
        return GeoStatus::interior;
    }

    //!@}
    //!@{
    //! \name Straight-line movement and boundary crossing

    /*!
     * Find the distance to the next boundary (infinite max).
     *
     * Determines the distance to the next boundary (i.e., a different
     * implementation volume) along the track's current direction.
     *
     * \deprecated Provide a physically reasonable upper bound to the distance
     * to reduce search cost and avoid a redundant method.
     */
    Propagation find_next_step()
    {
        return this->find_next_step(NumericLimits<real_type>::infinity());
    }

    /*!
     * Find the distance to the next boundary, up to and including a step.
     *
     * Determines the distance to the next boundary along the track's current
     * direction, up to a given distance. Queries may be more efficient for
     * small distances.
     *
     * \pre \c geo_status() is not \c GeoStatus::boundary_inc .
     * \post The returned distance is in the range \c (0, max_step] .
     */
    virtual Propagation find_next_step(real_type max_step) = 0;

    /*!
     * Move within the volume.
     *
     * Changes the physical position of the geometry state without altering
     * the logical state (i.e., it must remain within the current volume).
     *
     * \pre The given \c step must be less than or equal to the previous \c
     * find_next_step result and can only be equal if the endpoint is not on a
     * boundary.
     */
    virtual void move_internal(real_type step) = 0;

    /*!
     * Move to the boundary in preparation for crossing it.
     *
     * Moves the track to the boundary of the current volume along the current
     * direction, updating its logical state to indicate that it is on the
     * boundary of the current volume.
     *
     * \pre \c geo_status() is not \c GeoStatus::boundary_inc .
     * \post \c geo_status() is \c GeoStatus::boundary_inc .
     */
    virtual void move_to_boundary() = 0;

    /*!
     * Cross from one side of the current surface to the other.
     *
     * Changes the logical state when on the boundary, updating to the next
     * volume.
     *
     * \pre \c geo_status() is \c GeoStatus::boundary_inc .
     * \post \c geo_status() is \c GeoStatus::boundary_out , or \c
     *   GeoStatus::error if the new volume could not be found.
     */
    virtual void cross_boundary() = 0;
    //!@}
    //!@{
    //! \name Locally bounded movement

    /*!
     * Find the safety distance at the current position.
     *
     * Determines the distance to the nearest boundary in any direction (i.e.,
     * the radius of the maximally inscribed sphere).
     *
     * \c deprecated: use \c find_safety(inf)
     */
    virtual real_type find_safety() = 0;

    /*!
     * Find the safety at the current position, up to a maximum step distance.
     *
     * The resulting safety should be no larger than the maximum step.
     *
     * \pre \c is_on_boundary(geo.geo_status()) must be false
     */
    virtual real_type find_safety(real_type max_step) = 0;

    /*!
     * Change direction.
     *
     * Changes the direction of the track (in global coordinate system).
     */
    virtual void set_dir(Real3 const& newdir) = 0;

    /*!
     * Move within the volume to a specific point.
     *
     * Changes the physical position of the geometry state without altering
     * the logical state (i.e., it must remain within the current volume).
     */
    virtual void move_internal(Real3 const& pos) = 0;
    //!@}

  protected:
    GeoTrackInterface() = default;
    CELER_DEFAULT_COPY_MOVE(GeoTrackInterface);
};

//---------------------------------------------------------------------------//
#if CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT
extern template class GeoTrackInterface<float>;
#endif
extern template class GeoTrackInterface<double>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
