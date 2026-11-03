//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldObserver.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Ignore events in the field propagation loops.
 *
 * This is the default observer for \c FieldPropagator and \c
 * FieldSubstepper, and it defines the observer interface. Each member function
 * is called when the corresponding branch of the propagation loop is taken.
 * Because these functions are empty, the uninstrumented propagation code
 * compiles to the same thing as without an observer.
 *
 * Substepper events:
 * - \c quick_advance: a tiny step is integrated once without error control
 * - \c chord_iteration: a trial integration while searching for a chord whose
 *   sagitta is less than \c delta_chord
 * - \c accurate_advance: the chord's truncation error was too large, so the
 *   step is integrated again with adaptive error control
 * - \c accurate_iteration: an integration step inside the accurate advance
 * - \c good_step_reject: a trial integration was rejected because of its
 *   truncation error
 *
 * Propagator events, of which exactly one of the last four follows each
 * \c substep:
 * - \c substep: the substepper advanced the state
 * - \c substep_accepted: the chord did not intersect a boundary
 * - \c substep_bumped: a reentrant boundary was hit just after starting on a
 *   boundary, so the substep is halved
 * - \c substep_intercepted: the boundary intercept is close enough to the
 *   substep end to finish the step
 * - \c substep_shortened: the boundary intercept is too far from the substep
 *   end, so the substep is retried with a shorter length
 *
 * The propagator also calls \c stuck if the track failed to move and was
 * bumped.
 *
 * Observer member functions must be \c const because the substepper calls
 * them from \c const member functions. An observer that records events should
 * do so through a pointer or reference to external data.
 */
struct NullFieldObserver
{
    //!@{
    //! \name Substepper events
    CELER_FUNCTION void quick_advance() const {}
    CELER_FUNCTION void chord_iteration() const {}
    CELER_FUNCTION void accurate_advance() const {}
    CELER_FUNCTION void accurate_iteration() const {}
    CELER_FUNCTION void good_step_reject() const {}
    //!@}

    //!@{
    //! \name Propagator events
    CELER_FUNCTION void substep() const {}
    CELER_FUNCTION void substep_accepted() const {}
    CELER_FUNCTION void substep_bumped() const {}
    CELER_FUNCTION void substep_intercepted() const {}
    CELER_FUNCTION void substep_shortened() const {}
    CELER_FUNCTION void stuck() const {}
    //!@}
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
