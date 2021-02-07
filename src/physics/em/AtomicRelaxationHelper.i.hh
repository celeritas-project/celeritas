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
    const AtomicRelaxParamsPointers& shared,
    ElementId                        el_id,
    SecondaryAllocatorView&          allocate,
    size_type                        base_size)
    : shared_(shared), el_id_(el_id), allocate_(allocate), base_size_(base_size)
{
    CELER_EXPECT(!shared_ || el_id_ < shared_.elements.size());
}

//---------------------------------------------------------------------------//
/*!
 * Allocate space for secondaries
 */
CELER_FUNCTION Span<Secondary> AtomicRelaxationHelper::allocate() const
{
    size_type size = base_size_;
    if (*this)
    {
        // Maxiumum number of secondaries that could be produced in both
        // atomic relaxation and the primary process
        size += shared_.elements[el_id_.get()].max_secondary;
    }

    Secondary* secondaries = this->allocate_(size);
    if (secondaries == nullptr)
    {
        size = 0;
    }

    return {secondaries, size};
}

//---------------------------------------------------------------------------//
/*!
 * Create the sampling distribution from preallocated storage for secondaries
 * and the shell ID of the initial vacancy
 */
CELER_FUNCTION AtomicRelaxation AtomicRelaxationHelper::build_distribution(
    Span<Secondary> secondaries, SubshellId shell_id) const
{
    return AtomicRelaxation{shared_, el_id_, shell_id, secondaries, base_size_};
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
