//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/interactor/AtomicRelaxationHelper.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Span.hh"
#include "corecel/sys/ThreadId.hh"
#include "geocel/random/IsotropicDistribution.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/em/data/AtomicRelaxationData.hh"

#include "AtomicRelaxation.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for atomic relaxation.
 *
 * This class can be used inside an \c Interactor to simplify the creation of
 * the sampling distribution for relaxation and the allocation of storage for
 * secondaries created in both relaxation and the primary process.
 *
 * \code
    // Allocate secondaries for a model that produces a single secondary
    Span<Secondary> secondaries;
    size_type       count = relax_helper.max_secondaries() + 1;
    if (Secondary* ptr = allocate_secondaries(count))
    {
        secondaries = {ptr, count};
    }
    else
    {
        return Interaction::from_failure();
    }
    Interaction result;

    // ...

    AtomicRelaxation sample_relaxation = relax_helper.build_distribution(
        cutoffs, shell_id, secondaries.subspan(1));
    auto outgoing             = sample_relaxation(rng);
    result.secondaries        = outgoing.secondaries;
    result.energy_deposition -= outgoing.energy;
   \endcode
 */
class AtomicRelaxationHelper
{
  public:
    // Construct with the currently interacting element
    inline CELER_FUNCTION
    AtomicRelaxationHelper(AtomicRelaxParamsRef const& shared,
                           AtomicRelaxStateRef const& states,
                           ElementId el_id,
                           TrackSlotId tid);

    // Whether atomic relaxation should be applied
    explicit inline CELER_FUNCTION operator bool() const;

    // Space needed for relaxation secondaries
    inline CELER_FUNCTION size_type max_secondaries() const;

    // Storage for subshell ID stack
    inline CELER_FUNCTION Span<SubshellId> scratch() const;

    // Create the sampling distribution from sampled shell and allocated mem
    inline CELER_FUNCTION AtomicRelaxation
    build_distribution(CutoffView const& cutoffs,
                       SubshellId shell_id,
                       Span<Secondary> secondaries) const;

  private:
    AtomicRelaxParamsRef const& shared_;
    AtomicRelaxStateRef const& states_;
    ElementId const el_id_;
    TrackSlotId const track_slot_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
AtomicRelaxationHelper::AtomicRelaxationHelper(
    AtomicRelaxParamsRef const& shared,
    AtomicRelaxStateRef const& states,
    ElementId el_id,
    TrackSlotId tid)
    : shared_(shared), states_(states), el_id_(el_id), track_slot_(tid)
{
    CELER_EXPECT(!shared_ || el_id_ < shared_.elements.size());
    CELER_EXPECT(!states_ || track_slot_ < states.size());
    CELER_EXPECT(bool(shared_) == bool(states_));
}

//---------------------------------------------------------------------------//
/*!
 * Whether atomic relaxation should be applied.
 */
CELER_FUNCTION AtomicRelaxationHelper::operator bool() const
{
    // Atomic relaxation is enabled and the element has transition data
    return shared_ && shared_.elements[el_id_];
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of secondaries that can be produced.
 */
CELER_FUNCTION size_type AtomicRelaxationHelper::max_secondaries() const
{
    CELER_EXPECT(*this);
    return shared_.elements[el_id_].max_secondary;
}

//---------------------------------------------------------------------------//
/*!
 * Access scratch space.
 *
 * This temporary data is needed as part of a stack while processing the
 * cascade of electrons.
 */
CELER_FUNCTION Span<SubshellId> AtomicRelaxationHelper::scratch() const
{
    CELER_EXPECT(*this);
    auto offset = track_slot_.get() * shared_.max_stack_size;
    Span<SubshellId> all_scratch
        = states_.scratch[AllItems<SubshellId, MemSpace::native>{}];
    CELER_ENSURE(offset + shared_.max_stack_size <= all_scratch.size());
    return {all_scratch.data() + offset, shared_.max_stack_size};
}

//---------------------------------------------------------------------------//
/*!
 * Create the sampling distribution.
 */
CELER_FUNCTION AtomicRelaxation
AtomicRelaxationHelper::build_distribution(CutoffView const& cutoffs,
                                           SubshellId shell_id,
                                           Span<Secondary> secondaries) const
{
    CELER_EXPECT(*this);
    CELER_EXPECT(secondaries.size() == this->max_secondaries());
    return AtomicRelaxation{
        shared_, cutoffs, el_id_, shell_id, secondaries, this->scratch()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
