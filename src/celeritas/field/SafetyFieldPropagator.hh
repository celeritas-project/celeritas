//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/SafetyFieldPropagator.hh
//! \sa test/celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"
#include "corecel/math/ArrayOperators.hh"
#include "corecel/math/ArrayUtils.hh"
#include "geocel/SafetySphere.hh"
#include "geocel/Types.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Propagate a charged particle in a field inside a safety sphere.
 *
 * A safety sphere is a ball known to contain no geometry boundary (see \c
 * SafetySphere). If the track starts at \f$ \mathbf{x}_0 \f$ and the step
 * length is \f$ s \f$, the whole step must fit inside the sphere:
 * \f[
   |\mathbf{x}_0 - \mathbf{c}| + s + \delta \le r \,,
 * \f]
 * where \f$ \mathbf{c} \f$ and \f$ r \f$ are the sphere's center and radius
 * and \f$ \delta \f$ is the intersection tolerance. A track never moves
 * farther from its start than its path length, so every point it can reach
 * during the step is inside the sphere by at least \f$ \delta \f$ and no
 * boundary search is needed.
 *
 * Unlike \c FieldPropagator, this class never searches for boundaries: it only
 * integrates the field, and it updates the geometry state once at the end of
 * the step. Every boundary search \c FieldPropagator would make for such a
 * step finds no intersection, so the results are the same.
 *
 * Integration error could put a substep end slightly farther from the start
 * than the exact trajectory. If a substep end isn't inside the sphere by the
 * tolerance, the substep is retried with half the length.
 */
template<class SubstepperT, class GTV>
class SafetyFieldPropagator
{
  public:
    //!@{
    //! \name Type aliases
    using result_type = Propagation;
    //!@}

  public:
    // Construct with the field driver and a sphere containing the step
    inline CELER_FUNCTION SafetyFieldPropagator(
        SubstepperT&& advance,
        ParticleTrackView const& particle,
        GTV&& geo,
        SafetySphere const& sphere);

    // Move the track by a distance whose whole step fits inside the sphere
    inline CELER_FUNCTION result_type operator()(real_type step);

  private:
    //// DATA ////

    SubstepperT advance_;
    GTV geo_;
    SafetySphere sphere_;
    OdeState state_;
};

//---------------------------------------------------------------------------//
// DEDUCTION GUIDES
//---------------------------------------------------------------------------//
template<class SubstepperT, class GTV>
CELER_CTAD_FUNCTION SafetyFieldPropagator(
    SubstepperT&&, ParticleTrackView const&, GTV&&, SafetySphere const&)
    -> SafetyFieldPropagator<SubstepperT, GTV>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with the field driver and a sphere containing the step.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION SafetyFieldPropagator<SubstepperT, GTV>::SafetyFieldPropagator(
    SubstepperT&& advance,
    ParticleTrackView const& particle,
    GTV&& geo,
    SafetySphere const& sphere)
    : advance_(::celeritas::forward<SubstepperT>(advance))
    , geo_(::celeritas::forward<GTV>(geo))
    , sphere_(sphere)
{
    using MomentumUnits = OdeState::MomentumUnits;

    state_.pos = geo_.pos();
    state_.mom = value_as<MomentumUnits>(particle.momentum()) * geo_.dir();
}

//---------------------------------------------------------------------------//
/*!
 * Move the track by a distance whose whole step fits inside the sphere.
 *
 * The step ends early only if the maximum number of substeps is reached, in
 * which case the track is flagged as looping.
 */
template<class SubstepperT, class GTV>
CELER_FUNCTION auto SafetyFieldPropagator<SubstepperT, GTV>::operator()(
    real_type step) -> result_type
{
    real_type const tol = advance_.delta_intersection();
    CELER_EXPECT(step > 0);
    CELER_EXPECT(!geo_.is_on_boundary());
    CELER_EXPECT(sphere_.contains(state_.pos, step + tol));

    result_type result;
    result.boundary = false;
    result.distance = 0;

    real_type remaining = step;
    auto remaining_substeps = advance_.max_substeps();
    do
    {
        // Advance up to (but probably less than) the remaining step length
        Substep substep = advance_(remaining, state_);
        CELER_ASSERT(substep.length > 0 && substep.length <= remaining);

        if (CELER_UNLIKELY(!sphere_.contains(substep.state.pos, tol)))
        {
            // Integration error moved the end out of the sphere: retry a
            // shorter substep
            remaining = substep.length / 2;
            continue;
        }

        state_ = substep.state;
        result.distance += substep.length;
        remaining = step - result.distance;
        --remaining_substeps;
    } while (remaining > advance_.minimum_step() && remaining_substeps > 0);

    if (remaining_substeps == 0 && result.distance < step)
    {
        // Flag track as looping if the max number of substeps was reached
        // without moving the full step length
        result.looping = true;
    }
    else if (CELER_UNLIKELY(result.distance < step))
    {
        // The distance might be slightly less than the step due to roundoff
        // error: reset it so the track isn't marked as propagation-limited
        CELER_ASSERT(soft_equal(result.distance, step));
        result.distance = step;
    }

    // Update the geometry once, and conserve momentum through the field
    // change by setting the direction from the momentum
    geo_.move_internal(state_.pos);
    geo_.set_dir(make_unit_vector(state_.mom));

    CELER_ENSURE(
        result.distance > 0
        && (result.distance <= step || soft_equal(result.distance, step)));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
