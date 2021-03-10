//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocator.hh
//---------------------------------------------------------------------------//
#pragma once

#include "StackAllocatorInterface.hh"

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
    StackAllocator<Secondary> allocate;

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
       result.secondaries = Span<Secondary>{allocated, 2};
       return result;
    };
 };
 \endcode
 *
 * A later kernel could then iterate over the secondaries to apply cutoffs:
 * \code
 using SecondaryRef
     = StackAllocatorData<Secondary, Ownership::reference, MemSpace::device>;

 __global__ apply_cutoff(const SecondaryRef ptrs)
 {
     auto thread_idx = celeritas::KernelParamCalculator::thread_id().get();
     StackAllocator<Secondary> allocate(ptrs);
     auto secondaries = allocate.get();
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
 *
 * You *cannot* safely access the current size of the stack in the same kernel
 * that's modifying it -- if the stack attempts to allocate beyond the end,
 * then the \c size() call will reflect that overflowed state, rather than the
 * corrected size reflecting the failed allocation.
 *
 * A third kernel with a single thread would then be responsible for clearing
 * the data:
 * \code
 __global__ clear_stack(const SecondaryRef ptrs)
 {
     StackAllocator<Secondary> allocate(ptrs);
     auto thread_idx = celeritas::KernelParamCalculator::thread_id().get();
     if (thread_idx == 0)
     {
         allocate.clear();
     }
 }
 * \endcode
 *
 * These separate kernel launches are needed as grid-level synchronization
 * points.
 *
 * \todo Instead of returning a pointer, return IdRange<T>. Rename
 * StackAllocatorData to StackAllocation and have it look like a collection so
 * that *it* will provide access to the data. Better yet, have a
 * StackAllocation that can be a `const_reference` to the StackAllocatorData.
 * Then the rule will be "you can't create a StackAllocator in the same kernel
 * that you directly access a StackAllocation".
 */
template<class T>
class StackAllocator
{
  public:
    //!@{
    //! Type aliases
    using value_type  = T;
    using result_type = value_type*;
    using Pointers
        = StackAllocatorData<T, Ownership::reference, MemSpace::native>;
    //!@}

  public:
    // Construct with shared data
    explicit inline CELER_FUNCTION StackAllocator(const Pointers& data);

    // Total storage capacity (always safe)
    inline CELER_FUNCTION size_type capacity() const;

    //// INITIALIZE ////

    // Reset storage
    inline CELER_FUNCTION void clear();

    //// ALLOCATE ////

    // Allocate space for this many data
    inline CELER_FUNCTION result_type operator()(size_type count);

    //// ACCESS ////

    // Current size
    inline CELER_FUNCTION size_type size() const;

    // View all allocated data
    inline CELER_FUNCTION Span<value_type> get();
    inline CELER_FUNCTION Span<const value_type> get() const;

  private:
    const Pointers& data_;

    //// HELPER FUNCTIONS ////

    using SizeId    = ItemId<size_type>;
    using StorageId = ItemId<T>;
    static CELER_CONSTEXPR_FUNCTION SizeId size_id() { return SizeId{0}; }
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "StackAllocator.i.hh"
