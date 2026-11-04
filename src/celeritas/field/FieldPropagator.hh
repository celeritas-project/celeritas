//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/NumericLimits.hh"
#include "geocel/Types.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "Types.hh"

#include "detail/FieldUtils.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle in a field.
 *
 * For a given initial state (position, momentum), it propagates a charged
 * particle along a curved trajectory up to an interaction length proposed by
 * a chosen physics process for the step, possibly integrating sub-steps by
 * an adaptive step control with a required accuracy of tracking in a
 * field. It updates the final state (position, momentum, boundary) along with
 * the step actually taken.  If the final position is outside the current
 * volume, it returns a geometry limited step and the state at the
 * intersection between the curve trajectory and the first volume boundary
 * using an iterative step control method within a tolerance error imposed on
 * the closest distance between two positions by the field stepper and the
 * linear projection to the volume boundary.
 *
 * \note This follows similar methods to Geant4's G4PropagatorInField class.
 */
template<class SubstepperT, class GTV>
class FieldPropagator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = Propagation;
    //!@}

  public:
    // Construct with shared parameters and the field driver
    inline CELER_FUNCTION FieldPropagator(
        SubstepperT&& advance, ParticleTrackView const& particle, GTV&& geo);

    // Move track up to a user-provided distance, or to the next boundary
    inline CELER_FUNCTION result_type operator()(real_type dist);

  private:
    //// TYPES ////

    //! Progress along the requested step
    struct Progress
    {
        real_type step;  //!< Distance requested by the caller
        real_type remaining;  //!< Maximum length of the next substep
        short int remaining_substeps;  //!< Allowed substeps left
        result_type result;  //!< Accumulated result
    };

    //// HELPER FUNCTIONS ////

    // Start propagating over a step
    inline CELER_FUNCTION Progress start_step(real_type step) const;

    // Whether more substeps should be taken
    inline CELER_FUNCTION bool has_remaining(Progress const& p) const;

    // Search along the chord of a proposed substep and update progress
    inline CELER_FUNCTION void search_substep(Substep const&, Progress* p);

    // Update the geometry at the end of the step and return the result
    inline CELER_FUNCTION result_type finish_step(Progress const& p);

    //! Limit on substeps
    inline CELER_FUNCTION short int max_substeps() const;

    // Intersection tolerance
    inline CELER_FUNCTION real_type delta_intersection() const;

    // Distance to bump or to consider a "zero" movement
    inline CELER_FUNCTION real_type bump_distance() const;

    // Smallest allowable inner loop distance to take
    inline CELER_FUNCTION real_type minimum_substep() const;

  private:
    //// DATA ////

    SubstepperT advance_;
    GTV geo_;
    OdeState state_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class SubstepperT, class GTV>
CELER_CTAD_FUNCTION FieldPropagator(
    SubstepperT&&, ParticleTrackView const&, GTV&&)
    -> FieldPropagator<SubstepperT, GTV>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared field parameters and the field driver.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION FieldPropagator<SubstepperT, GTV>::FieldPropagator(
    SubstepperT&& advance, ParticleTrackView const& particle, GTV&& geo)
    : advance_(::celeritas::forward<SubstepperT>(advance))
    , geo_(::celeritas::forward<GTV>(geo))
{
    using MomentumUnits = OdeState::MomentumUnits;

    state_.pos = geo_.pos();
    state_.mom = value_as<MomentumUnits>(particle.momentum()) * geo_.dir();
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle in a field.
 *
 * It utilises a field driver (based on an adaptive step control to limit the
 * length traveled based on the magnetic field behavior and geometric
 * tolerances) to track a charged particle along a curved trajectory for a
 * given step length within a required accuracy or intersects
 * with a new volume (geometry limited step).
 *
 * The position of the internal OdeState `state_` should be consistent with the
 * geometry `geo_`'s position, but the geometry's direction will be a series
 * of "trial" directions that are the chords between the start and end points
 * of a curved substep through the field. At the end of the propagation step,
 * the geometry state's direction is updated based on the actual value of the
 * calculated momentum.
 *
 * Caveats:
 * - The physical (geometry track state) position may deviate from the exact
 *   curved propagation position up to a driver-based tolerance at every
 *   boundary crossing. The momentum will always be conserved, though.
 * - In some unusual cases (e.g. a very small caller-requested step, or an
 *   unusual accumulation in the driver's substeps) the distance returned may
 *   be slightly higher (again, up to a driver-based tolerance) than the
 *   physical distance travelled.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION auto FieldPropagator<SubstepperT, GTV>::operator()(
    real_type step) -> result_type
{
    CELER_EXPECT(step > 0);

    // Break the curved steps into substeps as determined by the driver *and*
    // by the proximity of geometry boundaries. Test for intersection with the
    // geometry boundary in each substep. This loop is guaranteed to converge
    // since the trial step always decreases *or* the actual position advances.
    Progress p = this->start_step(step);
    do
    {
        // Advance up to (but probably less than) the remaining step length
        this->search_substep(advance_(p.remaining, state_), &p);
    } while (this->has_remaining(p));

    return this->finish_step(p);
}

//---------------------------------------------------------------------------//
/*!
 * Start propagating over a step.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION auto FieldPropagator<SubstepperT, GTV>::start_step(
    real_type step) const -> Progress
{
    Progress p;
    p.step = step;
    p.remaining = step;
    p.remaining_substeps = this->max_substeps();
    p.result.boundary = geo_.is_on_boundary();
    p.result.distance = 0;
    return p;
}

//---------------------------------------------------------------------------//
/*!
 * Whether more substeps should be taken.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION bool FieldPropagator<SubstepperT, GTV>::has_remaining(
    Progress const& p) const
{
    return p.remaining > this->minimum_substep() && p.remaining_substeps > 0;
}

//---------------------------------------------------------------------------//
/*!
 * Search along the chord of a proposed substep and update progress.
 *
 * The substep is accepted if it doesn't intersect a boundary. Otherwise, it
 * either stops at the boundary or reduces the length of the next trial
 * substep.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION void FieldPropagator<SubstepperT, GTV>::search_substep(
    Substep const& substep, Progress* p)
{
    CELER_ASSERT(soft_zero(distance(state_.pos, geo_.pos())));
    CELER_ASSERT(p->result.boundary == geo_.is_on_boundary());
    CELER_ASSERT(substep.length > 0 && substep.length <= p->remaining);

    // Check whether the chord for this sub-step intersects a boundary
    auto chord = detail::make_chord(state_.pos, substep.state.pos);

    // Do a detailed boundary check from the start position toward
    // the substep end point.
    // Travel to the end of the chord, plus a little extra.
    if (chord.length >= this->minimum_substep())
    {
        // Only update the direction if the chord length is nontrivial.
        // This is usually the case but might be skipped in two cases:
        // - if the initial step is very small compared to the
        //   magnitude of the position (which can result in a zero length
        //   for the chord and NaNs for the direction)
        // - in a high-curvature track where the remaining distance is just
        //   barely above the remaining minimum step (in which case our
        //   boundary test does lose some accuracy)
        geo_.set_dir(chord.dir);
    }
    auto linear_step
        = geo_.find_next_step(chord.length + this->delta_intersection());

    // Scale the effective substep length to travel by the fraction along
    // the chord to the boundary. This value can be slightly larger than 1
    // because we search up a little past the endpoint (thanks to the delta
    // intersection).
    real_type const update_length = substep.length * linear_step.distance
                                    / chord.length;

    if (!linear_step.boundary)
    {
        // No boundary intersection along the chord: accept substep
        // movement inside the current volume and reset the remaining
        // distance so we can continue toward the next boundary or end of
        // caller-requested step. Reset the boundary flag to "false" only
        // in the unlikely case that we successfully shortened the substep
        // on a reentrant boundary crossing below.
        state_ = substep.state;
        p->result.boundary = false;
        p->result.distance += substep.length;
        p->remaining = p->step - p->result.distance;
        geo_.move_internal(state_.pos);
        --p->remaining_substeps;
    }
    else if (
        CELER_UNLIKELY(p->result.boundary
                       && ((linear_step.distance) < this->bump_distance())))
    {
        // Likely heading back into the old volume when starting on a
        // surface (this can happen when tracking through a volume at a
        // near tangent). Reduce substep size and try again.
        // TODO: this condition is triggered in ORANGE when on a reentrant
        // boundary
        p->remaining = substep.length / 2;
    }
    else if (update_length <= this->minimum_substep()
             || detail::is_intercept_close(state_.pos,
                                           chord.dir,
                                           linear_step.distance,
                                           substep.state.pos,
                                           this->delta_intersection())
             || chord.length == 0)
    {
        // We're close enough to the boundary that the next trial step
        // would be less than the driver's minimum step.
        // *OR*
        // The straight-line intersection point is a distance less than
        // `delta_intersection` from the substep's end position.
        // Commit the proposed state's momentum, use the
        // post-boundary-crossing track position for consistency, and
        // conservatively reduce the *reported* traveled distance to avoid
        // coincident boundary crossings.

        // Only cross the boundary if the intersect point is less
        // than or exactly on the boundary, or if the crossing
        // doesn't put us past the end of the step
        p->result.boundary = (linear_step.distance <= chord.length
                              || p->result.distance + update_length <= p->step
                              || chord.length == 0);

        if (!p->result.boundary)
        {
            // Don't move to the boundary, but instead move to the end of
            // the substep. This should result in basically the same effect
            // as "!linear_step.boundary" above.
            state_.pos = substep.state.pos;
            geo_.move_internal(substep.state.pos);
        }

        // The update length can be slightly greater than the substep due
        // to the extra delta_intersection boost when searching. The
        // substep itself can be more than the requested step.
        p->result.distance += celeritas::min(update_length, substep.length);
        state_.mom = substep.state.mom;
        p->remaining = 0;
    }
    else
    {
        // The straight-line intercept is too far from substep's end state.
        // Decrease the allowed substep (curved path distance) by the
        // fraction along the chord, and retry the driver step.
        p->remaining = update_length;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Update the geometry at the end of the step and return the result.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION auto FieldPropagator<SubstepperT, GTV>::finish_step(
    Progress const& p) -> result_type
{
    result_type result = p.result;

    if (p.remaining_substeps == 0 && result.distance < p.step)
    {
        // Flag track as looping if the max number of substeps was reached
        // without hitting a boundary or moving the full step length
        result.looping = true;
    }
    else if (result.distance > 0)
    {
        if (result.boundary)
        {
            // We moved to a new boundary. Update the position to reflect the
            // geometry's state (and possibly "bump" the ODE state's position
            // because of the tolerance in the intercept checks above).
            geo_.move_to_boundary();
            state_.pos = geo_.pos();
        }
        else if (CELER_UNLIKELY(result.distance < p.step))
        {
            // Even though the track traveled the full step length, the
            // distance might be slightly less than the step due to roundoff
            // error. Reset the distance so the track's action isn't
            // erroneously set as propagation-limited.
            CELER_ASSERT(soft_equal(result.distance, p.step));
            result.distance = p.step;
        }
    }

    // Even though the along-substep movement was through chord lengths,
    // conserve momentum through the field change by updating the final
    // *direction* based on the state's momentum.
    Real3 dir = make_unit_vector(state_.mom);
    geo_.set_dir(dir);

    if (CELER_UNLIKELY(result.distance == 0))
    {
        // We failed to move at all, which means we hit a boundary no matter
        // what step length we took, which means we're stuck.
        // Using the just-reapplied direction, hope that we're pointing deeper
        // into the current volume and bump the particle.
        // TODO: move this into a higher-level failure/bump mechanic
        result.distance = celeritas::min(this->bump_distance(), p.step);
        result.boundary = false;
        axpy(result.distance, dir, &state_.pos);
        geo_.move_internal(state_.pos);
    }
    else
    {
        CELER_ENSURE(result.boundary == geo_.is_on_boundary());
    }

    // Due to accumulation errors from multiple substeps or chord-finding
    // within the driver, the distance may be very slightly beyond the
    // requested step.
    CELER_ENSURE(
        (result.distance > 0 || geo_.failed())
        && (result.distance <= p.step || soft_equal(result.distance, p.step)));
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Distance close enough to a boundary to mark as being on the boundary.
 *
 * TODO: change delta intersection from property in FieldDriverOptions to
 * another FieldPropagatorOptions
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION real_type
FieldPropagator<SubstepperT, GTV>::delta_intersection() const
{
    return advance_.delta_intersection();
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of substeps.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION short int FieldPropagator<SubstepperT, GTV>::max_substeps() const
{
    return advance_.max_substeps();
}

//---------------------------------------------------------------------------//
/*!
 * Distance to bump or to consider a "zero" movement.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION real_type
FieldPropagator<SubstepperT, GTV>::minimum_substep() const
{
    return advance_.minimum_step();
}

//---------------------------------------------------------------------------//
/*!
 * Distance to bump or to consider a "zero" movement.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION real_type FieldPropagator<SubstepperT, GTV>::bump_distance() const
{
    using namespace celeritas::literals;

    return this->delta_intersection() * 0.1_r;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
