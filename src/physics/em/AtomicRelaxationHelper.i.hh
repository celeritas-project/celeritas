//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
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
    const AtomicRelaxParamsPointers& shared, ElementId el_id)
    : shared_(shared), el_id_(el_id)
{
    CELER_EXPECT(!shared_ || el_id_ < shared_.elements.size());
}

//---------------------------------------------------------------------------//
/*!
 * Whether atomic relaxation should be applied.
 */
CELER_FUNCTION AtomicRelaxationHelper::operator bool() const
{
    // Atomic relaxation is enabled and the element has transition data
    return shared_ && shared_.elements[el_id_.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of secondaries that can be produced.
 */
CELER_FUNCTION size_type AtomicRelaxationHelper::max_secondaries() const
{
    CELER_EXPECT(*this);
    return shared_.elements[el_id_.get()].max_secondary;
}

//---------------------------------------------------------------------------//
/*!
 * Maximum number of unprocessed subshell vacancies.
 *
 * This temporary data is needed as part of a stack while processing the
 * cascade of electrons.
 */
CELER_FUNCTION size_type AtomicRelaxationHelper::max_vacancies() const
{
    CELER_EXPECT(*this);
    return shared_.elements[el_id_.get()].max_stack_size;
}

//---------------------------------------------------------------------------//
/*!
 * Create the sampling distribution.
 */
CELER_FUNCTION AtomicRelaxation
AtomicRelaxationHelper::build_distribution(SubshellId       shell_id,
                                           Span<Secondary>  secondaries,
                                           Span<SubshellId> vacancies) const
{
    CELER_EXPECT(*this);
    CELER_EXPECT(secondaries.size() == this->max_secondaries());
    CELER_EXPECT(vacancies.size() == this->max_vacancies());
    return AtomicRelaxation{shared_, el_id_, shell_id, secondaries, vacancies};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
