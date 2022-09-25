//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iostream>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/ArrayIO.hh"
#include "corecel/io/ColorUtils.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "orange/Types.hh"
#include "celeritas/geo/GeoTrackView.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "Types.hh"
#include "detail/FieldUtils.hh"
using std::cout;
using std::endl;

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
 * \note This follows similar methods as in Geant4's G4PropagatorInField class.
 */
template<class DriverT>
class FieldPropagator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = Propagation;
    //!@}

  public:
    // Construct with shared parameters and the field driver
    inline CELER_FUNCTION FieldPropagator(DriverT&& driver,
                                          ParticleTrackView const& particle,
                                          GeoTrackView* geo);

    // Move track to next volume boundary.
    inline CELER_FUNCTION result_type operator()();

    // Move track up to a user-provided distance, or to the next boundary
    inline CELER_FUNCTION result_type operator()(real_type dist);

    //! Limit on substeps
    static CELER_CONSTEXPR_FUNCTION short int max_substeps() { return 128; }

    // Distance to bump or to consider a "zero" movement
    inline CELER_FUNCTION real_type bump_distance() const;

  private:
    //// DATA ////

    DriverT driver_;
    GeoTrackView& geo_;
    OdeState state_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared field parameters and the field driver.
 */
template<class DriverT>
CELER_FUNCTION
FieldPropagator<DriverT>::FieldPropagator(DriverT&& driver,
                                          ParticleTrackView const& particle,
                                          GeoTrackView* geo)
    : driver_(::celeritas::forward<DriverT>(driver)), geo_(*geo)
{
    CELER_ASSERT(geo);

    using MomentumUnits = OdeState::MomentumUnits;

    state_.pos = geo_.pos();
    state_.mom
        = detail::ax(value_as<MomentumUnits>(particle.momentum()), geo_.dir());
}

//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle until it hits a boundary.
 */
template<class DriverT>
CELER_FUNCTION auto FieldPropagator<DriverT>::operator()() -> result_type
{
    return (*this)(numeric_limits<real_type>::infinity());
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
template<class DriverT>
CELER_FUNCTION auto FieldPropagator<DriverT>::operator()(real_type step)
    -> result_type
{
    CELER_EXPECT(step > 0);
    result_type result;
    result.boundary = geo_.is_on_boundary();
    result.distance = 0;

    cout << color_code('b') << "Propagate up to " << step << color_code(' ')
         << endl;

    // Break the curved steps into substeps as determined by the driver *and*
    // by the proximity of geometry boundaries. Test for intersection with the
    // geometry boundary in each substep. This loop is guaranteed to converge
    // since the trial step always decreases *or* the actual position advances.
    real_type remaining = step;
    auto remaining_substeps = this->max_substeps();
    do
    {
        CELER_ASSERT(soft_zero(distance(state_.pos, geo_.pos())));
        CELER_ASSERT(result.boundary == geo_.is_on_boundary());

        // Advance up to (but probably less than) the remaining step length
        DriverResult substep = driver_.advance(remaining, state_);
        CELER_ASSERT(substep.step <= remaining
                     || soft_equal(substep.step, remaining));

        cout << "- advance(" << remaining << ", " << state_.pos << ") -> {"
             << substep.step << ", " << substep.state.pos << "}" << endl;

        // Check whether the chord for this sub-step intersects a boundary
        auto chord = detail::make_chord(state_.pos, substep.state.pos);

        // Do a detailed check boundary check from the start position toward
        // the substep end point. Travel to the end of the chord, plus a little
        // extra.
        if (chord.length >= driver_.minimum_step())
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
            = geo_.find_next_step(chord.length + driver_.delta_intersection());

        cout << " + chord length " << chord.length << " => linear step "
             << linear_step.distance;
        if (linear_step.boundary)
        {
            cout << " (hit surface " << geo_.next_surface_id().unchecked_get()
                 << ')';
        }
        cout << '\n';

        if (!linear_step.boundary)
        {
            // No boundary intersection along the chord: accept substep
            // movement inside the current volume and reset the remaining
            // distance so we can continue toward the next boundary or end of
            // caller-requested step. Reset the boundary flag to "false" only
            // in the unlikely case that we successfully shortened the substep
            // on a reentrant boundary crossing below.
            state_ = substep.state;
            result.boundary = false;
            result.distance += celeritas::min(substep.step, remaining);
            remaining = step - result.distance;
            geo_.move_internal(state_.pos);
            --remaining_substeps;
            cout << " + advancing to substep end point (" << remaining_substeps
                 << " remaining)" << endl;
        }
        else if (CELER_UNLIKELY(result.boundary
                                && linear_step.distance < this->bump_distance()))
        {
            // Likely heading back into the old volume when starting on a
            // surface (this can happen when tracking through a volume at a
            // near tangent). Reduce substep size and try again. Assume a
            // boundary crossing if repeated bisection of the substep fails to
            // converge.
            remaining = substep.step / 2;
            cout << " + halving substep distance" << endl;
        }
        else if (substep.step * linear_step.distance
                 <= driver_.minimum_step() * chord.length)
        {
            // i.e.: substep * (linear_step / chord_length) <= min_step
            // We're close enough to the boundary that the next trial step
            // would be less than the driver's minimum step. Accept the
            // momentum update, but use the position from the new boundary.
            result.boundary = true;
            result.distance += min(linear_step.distance, remaining);
            state_.mom = substep.state.mom;
            remaining = 0;
            cout << " + next trial step exceeds driver minimum "
                 << driver_.minimum_step() << endl;
        }
        else if (detail::is_intercept_close(state_.pos,
                                            chord.dir,
                                            linear_step.distance,
                                            substep.state.pos,
                                            driver_.delta_intersection()))
        {
            // The straight-line intersection point is a distance less than
            // `delta_intersection` from the substep's end position.
            // Commit the proposed state's momentum, use the
            // post-boundary-crossing track position for consistency, and
            // conservatively reduce the *reported* traveled distance to avoid
            // coincident boundary crossings.
            result.boundary = true;
            real_type miss_distance = std::sqrt(detail::calc_miss_distance_sq(
                state_.pos, chord.dir, linear_step.distance, substep.state.pos));
            CELER_ASSERT(miss_distance >= 0
                         && miss_distance <= driver_.delta_intersection());
            result.distance += substep.step - miss_distance;
            state_.mom = substep.state.mom;
            remaining = 0;

            cout << " + intercept is sufficiently close (miss distance = "
                 << miss_distance << ") to substep point" << endl;
        }
        else
        {
            // The straight-line intercept is too far from substep's end state.
            // Decrease the allowed substep (curved path distance) by the
            // fraction along the chord, and retry the driver step.
            remaining = substep.step * linear_step.distance / chord.length;
            cout << " + Setting remaining distance to a fraction "
                 << linear_step.distance / chord.length << " of the substep"
                 << endl;
        }
    } while (remaining >= driver_.minimum_step() && remaining_substeps > 0);

    if (result.distance > 0)
    {
        if (result.boundary)
        {
            // We moved to a new boundary. Update the position to reflect the
            // geometry's state (and possibly "bump" the ODE state's position
            // because of the tolerance in the intercept checks above).
            geo_.move_to_boundary();
            state_.pos = geo_.pos();
            cout << "- Moved to boundary " << geo_.surface_id().unchecked_get()
                 << " at position " << state_.pos << endl;
        }
        else if (remaining_substeps > 0)
        {
            // Bad luck with substep accumulation or possible very small
            // initial value for "step". Return that we've moved this tiny
            // amount (for, e.g., dE/dx purposes) but don't physically
            // propagate the track.
            result.distance += remaining;
            cout << "- Moved distance " << remaining
                 << " without physically changing position" << endl;
        }
    }

    // Even though the along-substep movement was through chord lengths,
    // conserve momentum through the field change by updating the final
    // *direction* based on the state's momentum.
    Real3 dir = state_.mom;
    normalize_direction(&dir);
    geo_.set_dir(dir);

    if (CELER_UNLIKELY(result.distance == 0))
    {
        // We failed to move at all, which means we hit a boundary no matter
        // what step length we took, which means we're stuck.
        // Using the just-reapplied direction, hope that we're pointing deeper
        // into the current volume and bump the particle.
        result.distance = celeritas::min(this->bump_distance(), step);
        result.boundary = false;
        axpy(result.distance, dir, &state_.pos);
        geo_.move_internal(state_.pos);
    }

    cout << color_code('g') << "==> distance " << result.distance
         << color_code(' ') << " (in "
         << this->max_substeps() - remaining_substeps << " steps)" << endl;

    CELER_ENSURE(result.boundary == geo_.is_on_boundary());
    CELER_ENSURE(result.distance > 0 && result.distance <= step);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Distance to bump or to consider a "zero" movement.
 *
 * Currently this is set to the field driver's minimum step, but it should
 * probably be related to the geometry instead.
 */
template<class DriverT>
CELER_FUNCTION real_type FieldPropagator<DriverT>::bump_distance() const
{
    return driver_.minimum_step();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
