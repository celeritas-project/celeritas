//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldPropagationCounters.hh
//! \sa test/celeritas/field/FieldPropagator.test.cc
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Number of events during a single field propagation call.
 *
 * These are filled by \c CountingFieldObserver and by the counting
 * integrator and geometry wrappers, and they characterize the cost of a
 * propagation step:
 * - Each \c substep is one call to the substepper, followed by one boundary
 *   search (\c intersects) and exactly one of the \c accepted, \c bumped,
 *   \c intercepted, or \c shortened outcomes.
 * - Each \c integrations is one evaluation of the integrator (several field
 *   evaluations), and it is the sum of the quick-advance, chord-search, and
 *   accurate-advance integrations (including rejected good steps).
 * - \c first_substep_integrations is the number of integrations used by the
 *   first substep. Because the substepper's chord estimate is not saved
 *   between propagation calls, the first substep usually starts with the full
 *   step length.
 * - \c skippable_intersects is the number of boundary searches whose search
 *   segment lies entirely inside a safety sphere at the start of the step. A
 *   propagator that used that safety sphere could skip those searches.
 */
struct FieldPropagationCounters
{
    //!@{
    //! \name Propagator loop
    size_type substeps{0};
    size_type accepted{0};
    size_type bumped{0};
    size_type intercepted{0};
    size_type shortened{0};
    size_type stuck{0};
    //!@}

    //!@{
    //! \name Substepper loops
    size_type integrations{0};
    size_type first_substep_integrations{0};
    size_type quick_advances{0};
    size_type chord_iterations{0};
    size_type accurate_advances{0};
    size_type accurate_iterations{0};
    size_type good_step_rejects{0};
    //!@}

    //!@{
    //! \name Geometry navigation
    size_type intersects{0};
    size_type intersect_hits{0};
    size_type skippable_intersects{0};
    size_type safeties{0};
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Count events in the field propagation loops.
 *
 * This implements the \c NullFieldObserver interface by incrementing the
 * corresponding counter of an externally owned \c FieldPropagationCounters.
 * Since the substepper and propagator each store a copy of the observer, it
 * holds a pointer to the counters.
 */
class CountingFieldObserver
{
  public:
    //! Construct with a pointer to the counters to increment
    explicit CELER_FUNCTION CountingFieldObserver(
        FieldPropagationCounters* counters)
        : c_{counters}
    {
        CELER_EXPECT(c_);
    }

    //!@{
    //! \name Substepper events
    CELER_FUNCTION void quick_advance() const { ++c_->quick_advances; }
    CELER_FUNCTION void chord_iteration() const { ++c_->chord_iterations; }
    CELER_FUNCTION void accurate_advance() const { ++c_->accurate_advances; }
    CELER_FUNCTION void accurate_iteration() const
    {
        ++c_->accurate_iterations;
    }
    CELER_FUNCTION void good_step_reject() const { ++c_->good_step_rejects; }
    //!@}

    //!@{
    //! \name Propagator events
    inline CELER_FUNCTION void substep() const;
    CELER_FUNCTION void substep_accepted() const { ++c_->accepted; }
    CELER_FUNCTION void substep_bumped() const { ++c_->bumped; }
    CELER_FUNCTION void substep_intercepted() const { ++c_->intercepted; }
    CELER_FUNCTION void substep_shortened() const { ++c_->shortened; }
    CELER_FUNCTION void stuck() const { ++c_->stuck; }
    //!@}

  private:
    FieldPropagationCounters* c_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Count a substep and save the integrations taken by the first one.
 *
 * The substepper has already called the integrator for this substep, so the
 * integration counter includes it.
 */
CELER_FUNCTION void CountingFieldObserver::substep() const
{
    if (c_->substeps == 0)
    {
        c_->first_substep_integrations = c_->integrations;
    }
    ++c_->substeps;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
