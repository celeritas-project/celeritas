//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Macros.hh"
#include "StackAllocatorPointers.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Low-level functor for allocating memory on a stack.
 *
 * This class acts like \c malloc but with no alignment. The calling class
 * should make sure each allocation is sufficient for \c alignof(T) or 16. It
 * performs no initialization on the data.
 *
 * \code
    StackAllocator alloc(view);
    double* temp_dbl = alloc(4 * sizeof(double));
   \endcode
 */
class StackAllocator
{
  public:
    //@{
    //! Type aliases
    using result_type = void*;
    using size_type   = StackAllocatorPointers::size_type;
    //@}

  public:
    // Construct with a reference to the storage pointers
    explicit inline CELER_FUNCTION
    StackAllocator(const StackAllocatorPointers&);

    // Allocate like malloc
    inline CELER_FUNCTION result_type operator()(size_type size);

  private:
    const StackAllocatorPointers& shared_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "StackAllocator.i.hh"

//---------------------------------------------------------------------------//
