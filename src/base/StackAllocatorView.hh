//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "StackAllocatorInterface.hh"
#include "Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Dynamically allocate arbitrary data on a stack.
 *
 * The stack allocator view acts as a functor and accessor to the allocated
 * data. It enables very fast on-device dynamic allocation of data, such as
 * secondaries or detector hits. As an example, inside a hypothetical physics
 * Interactor class, you could create two particles with the following code:
 * \code
 struct Interactor
 {
    StackAllocatorView<Secondary> allocate; //!< Mutable allocator view

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
       result.secondaries = Span<Secondary>{allocated, 4};
       return result;
    };
 };
 \endcode
 * A later kernel could then iterate over the secondaries to apply cutoffs:
 * \code
   __global__ apply_cutoff(const StackAllocatorPointers<Secondary> ptrs)
   {
       auto thread_idx = celeritas::KernelParamCalculator::thread_id().get();
       StackAllocatorView<Secondary> view(ptrs);
       auto secondaries = view.get();
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
template<class T>
class StackAllocatorView
{
  public:
    //!@{
    //! Type aliases
    using value_type  = T;
    using result_type = value_type*;
    using Pointers    = StackAllocatorPointers<T>;
    using size_type   = typename Pointers::size_type;
    //!@}

  public:
    // Construct with shared data
    explicit inline CELER_FUNCTION StackAllocatorView(const Pointers& shared);

    // Allocate space for this many data
    inline CELER_FUNCTION result_type operator()(size_type count);

    // Total storage capacity
    inline CELER_FUNCTION size_type capacity() const;

    // View all allocated data
    inline CELER_FUNCTION Span<value_type> get();
    inline CELER_FUNCTION Span<const value_type> get() const;

  private:
    const Pointers& shared_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "StackAllocatorView.i.hh"
