//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file AtomicRelaxationHelper.i.hh
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION
AtomicRelaxationHelper::AtomicRelaxationHelper(
    const AtomicRelaxParamsRef& shared,
    const AtomicRelaxStateRef&  states,
    ElementId                   el_id,
    ThreadId                    tid)
    : shared_(shared), states_(states), el_id_(el_id), thread_(tid)
{
    CELER_EXPECT(!shared_ || el_id_ < shared_.elements.size());
    CELER_EXPECT(!states_ || thread_ < states.size());
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
    auto             offset = thread_.get() * shared_.max_stack_size;
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
AtomicRelaxationHelper::build_distribution(const CutoffView& cutoffs,
                                           SubshellId        shell_id,
                                           Span<Secondary>   secondaries) const
{
    CELER_EXPECT(*this);
    CELER_EXPECT(secondaries.size() == this->max_secondaries());
    return AtomicRelaxation{
        shared_, cutoffs, el_id_, shell_id, secondaries, this->scratch()};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
