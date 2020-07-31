//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file SecondaryAllocatorView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Secondary.hh"
#include "SecondaryAllocatorPointers.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Stack of secondary particles added by a physics process.
 *
 * The secondary allocator view acts as a functor and accessor to the allocated
 * data. As an example, inside a hypothetical physics Interactor class, you
 * could create two particles with the following code:
 * \code
 struct Interactor
 {
    SecondaryAllocatorView allocate; //!< Mutable allocator view

    // Sample an interaction
    template<class Engine>
    Interaction operator()(Engine&)
    {
       // Create 2 secondary particles
       Secondary* allocated = this->allocate(2);
       if (!allocated)
       {
           return Interaction::from_failure();
       }
       Interaction result;
       result.secondaries = span<Secondary>{allocated, 4};
       return result;
    };
 };
 \endcode
 * A later kernel could then iterate over the secondaries to apply cutoffs:
 * \code
   __global__ apply_cutoff(const SecondaryAllocatorPointers ptrs)
   {
       auto thread_idx = celeritas::KernelParamCalculator::thread_id().get();
       SecondaryAllocatorView view(ptrs);
       auto secondaries = view.secondaries();
       if (thread_idx < secondaries.size())
       {
           Secondary& sec = secondaries[thread_idx];
           if (sec.energy < 100 * units::kilo_electron_volts)
           {
               sec.energy = 0;
           }
       }
   }
 * \endcode
 */
class SecondaryAllocatorView
{
  public:
    //@{
    //! Type aliases
    using size_type          = SecondaryAllocatorPointers::size_type;
    using result_type        = Secondary*;
    using constSpanSecondary = span<const Secondary>;
    using SpanSecondary      = span<Secondary>;
    //@}

  public:
    // Construct with shared data
    explicit inline CELER_FUNCTION
    SecondaryAllocatorView(const SecondaryAllocatorPointers& shared);

    // Allocate space for this many secondaries
    inline CELER_FUNCTION result_type operator()(size_type count);

    // Total storage capacity
    inline CELER_FUNCTION size_type capacity() const;

    // View all active secondaries
    inline CELER_FUNCTION SpanSecondary secondaries();

    // View all active secondaries
    inline CELER_FUNCTION constSpanSecondary secondaries() const;

  private:
    const SecondaryAllocatorPointers& shared_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "SecondaryAllocatorView.i.hh"
