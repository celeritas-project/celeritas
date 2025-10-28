//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/StackAllocator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <new>

#include "corecel/math/Atomics.hh"

#include "StackAllocatorData.hh"

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
    //! \name Type aliases
    using value_type = T;
    using result_type = value_type*;
    using Data = StackAllocatorData<T, Ownership::reference, MemSpace::native>;
    //!@}

  public:
    // Construct with shared data
    explicit inline CELER_FUNCTION StackAllocator(Data const& data);

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
    inline CELER_FUNCTION Span<value_type const> get() const;

  private:
    Data const& data_;

    //// HELPER FUNCTIONS ////

    using SizeId = ItemId<size_type>;
    using StorageId = ItemId<T>;
    static CELER_CONSTEXPR_FUNCTION SizeId size_id() { return SizeId{0}; }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<class T>
CELER_FUNCTION StackAllocator<T>::StackAllocator(Data const& shared)
    : data_(shared)
{
    CELER_EXPECT(shared);
}

//---------------------------------------------------------------------------//
/*!
 * Get the maximum number of values that can be allocated.
 */
template<class T>
CELER_FUNCTION auto StackAllocator<T>::capacity() const -> size_type
{
    return data_.storage.size();
}

//---------------------------------------------------------------------------//
/*!
 * Clear the stack allocator.
 *
 * This sets the size to zero. It should ideally *only* be called by a single
 * thread (though multiple threads resetting it should also be OK), but
 * *cannot be used in the same kernel that is allocating or viewing it*. This
 * is because the access times between different threads or thread-blocks is
 * indeterminate inside of a single kernel.
 */
template<class T>
CELER_FUNCTION void StackAllocator<T>::clear()
{
    data_.size[this->size_id()] = 0;
}

//---------------------------------------------------------------------------//
/*!
 * Allocate space for a given number of items.
 *
 * Returns NULL if allocation failed due to out-of-memory. Ensures that the
 * shared size reflects the amount of data allocated.
 */
template<class T>
CELER_FUNCTION auto StackAllocator<T>::operator()(size_type count)
    -> result_type
{
    CELER_EXPECT(count > 0);

    // Atomic add 'count' to the shared size
    size_type start = atomic_add(&data_.size[this->size_id()], count);
    if (CELER_UNLIKELY(start + count > data_.storage.size()))
    {
        // Out of memory: restore the old value so that another thread can
        // potentially use it. Multiple threads are likely to exceed the
        // capacity simultaneously. Only one has a "start" value less than or
        // equal to the total capacity: the remainder are (arbitrarily) higher
        // than that.
        if (start <= this->capacity())
        {
            // We were the first thread to exceed capacity, even though other
            // threads might have failed (and might still be failing) to
            // allocate. Restore the actual allocated size to the start value.
            // This might allow another thread with a smaller allocation to
            // succeed, but it also guarantees that at the end of the kernel,
            // the size reflects the actual capacity.
            data_.size[this->size_id()] = start;
        }

        /*!
         * \todo It might be useful to set an "out of memory" flag to make it
         * easier for host code to detect whether a failure occurred, rather
         * than looping through primaries and testing for failure.
         */

        // Return null pointer, indicating failure to allocate.
        return nullptr;
    }

    // Initialize the data at the newly "allocated" address
    value_type* result = &data_.storage[StorageId{start}];
    for (size_type i = 0; i < count; ++i)
    {
        result[i] = value_type{};
    }
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the number of items currently present.
 *
 * This value may not be meaningful (may be less than "actual" size) if
 * called in the same kernel as other threads that are allocating.
 */
template<class T>
CELER_FUNCTION auto StackAllocator<T>::size() const -> size_type
{
    size_type result = data_.size[this->size_id()];
    CELER_ENSURE(result <= this->capacity());
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * View all allocated data.
 *
 * This cannot be called while any running kernel could be modifying the size.
 */
template<class T>
CELER_FUNCTION auto StackAllocator<T>::get() -> Span<value_type>
{
    return data_.storage[ItemRange<T>{StorageId{0}, StorageId{this->size()}}];
}

//---------------------------------------------------------------------------//
/*!
 * View all allocated data (const).
 *
 * This cannot be called while any running kernel could be modifying the size.
 */
template<class T>
CELER_FUNCTION auto StackAllocator<T>::get() const -> Span<value_type const>
{
    return data_.storage[ItemRange<T>{StorageId{0}, StorageId{this->size()}}];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
