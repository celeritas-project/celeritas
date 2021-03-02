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
    const AtomicRelaxParamsPointers&   shared,
    const SubshellIdAllocatorPointers& vacancies,
    ElementId                          el_id,
    SecondaryAllocatorView&            allocate,
    size_type                          base_size)
    : shared_(shared)
    , vacancies_(vacancies)
    , el_id_(el_id)
    , allocate_(allocate)
    , base_size_(base_size)
{
    CELER_EXPECT(!shared_ || el_id_ < shared_.elements.size());
}

//---------------------------------------------------------------------------//
/*!
 * Allocate space for secondaries
 */
CELER_FUNCTION
Span<Secondary> AtomicRelaxationHelper::allocate_secondaries() const
{
    // Maxiumum number of secondaries that could be produced in both atomic
    // relaxation and the primary process
    size_type  size        = base_size_;
    Secondary* secondaries = nullptr;
    if (*this)
    {
        size += shared_.elements[el_id_.get()].max_secondary;
    }

    if (size > 0)
    {
        secondaries = allocate_(size);
        if (secondaries == nullptr)
            size = 0;
    }

    return {secondaries, size};
}

//---------------------------------------------------------------------------//
/*!
 * Allocate space for the unprocessed subshell vacancies
 */
CELER_FUNCTION
Span<SubshellId> AtomicRelaxationHelper::allocate_vacancies() const
{
    size_type   size      = 0;
    SubshellId* vacancies = nullptr;
    if (*this)
    {
        CELER_ASSERT(vacancies_);
        SubshellIdAllocatorView allocate(vacancies_);
        size      = shared_.elements[el_id_.get()].max_stack_size;
        vacancies = allocate(size);
        if (vacancies == nullptr)
            size = 0;
    }

    return {vacancies, size};
}

//---------------------------------------------------------------------------//
/*!
 * Create the sampling distribution from preallocated storage for secondaries
 * and vacancies and the shell ID of the initial vacancy
 */
CELER_FUNCTION AtomicRelaxation
AtomicRelaxationHelper::build_distribution(Span<Secondary>  secondaries,
                                           Span<SubshellId> vacancies,
                                           SubshellId       shell_id) const
{
    return AtomicRelaxation{
        shared_, el_id_, shell_id, secondaries, vacancies, base_size_};
}

//---------------------------------------------------------------------------//
/*!
 * Whether atomic relaxation should be applied
 */
CELER_FUNCTION AtomicRelaxationHelper::operator bool() const
{
    // Atomic relaxation is enabled and the element has transition data
    return shared_ && shared_.elements[el_id_.get()];
}

//---------------------------------------------------------------------------//
} // namespace celeritas
