//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorContainer.cu
//---------------------------------------------------------------------------//
#include "StackAllocatorContainer.hh"

#include <thrust/device_vector.h>
#include "Assert.hh"
#include "Macros.hh"
#include "base/KernelParamCalculator.cuda.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
struct StackAllocatorContainerPimpl
{
    thrust::device_vector<char> storage;
};

//---------------------------------------------------------------------------//
/*!
 * Reset the stack's state.
 *
 * This only clears the size counter; it doesn't reset the memory. Downstream
 * containers should take responsibility for that.
 */
__global__ void stack_allocator_clear_impl(StackAllocatorView data)
{
    auto local_thread_id = KernelParamCalculator::thread_id();
    if (local_thread_id.get() == 0)
    {
        *data.size = 0;
    }
}

//---------------------------------------------------------------------------//
/*!
 * Construct with the number of bytes to allocate on device.
 */
StackAllocatorContainer::StackAllocatorContainer(size_type capacity)
    : capacity_(capacity)
{
    REQUIRE(capacity > 0 && capacity % 16 == 0);

    // Round capacity up to nearest 'minimum alignment' requirement, and add
    // storage for on-device 'size' variable
    size_type alloc_size = capacity_;
    alloc_size += sizeof(size_type);

    // Allocate and copy data to device
    data_ = std::make_unique<StackAllocatorContainerPimpl>();
    data_->storage.resize(alloc_size + sizeof(size_type));
    this->clear();

    ENSURE(data_->storage.size() >= this->capacity());
}

//---------------------------------------------------------------------------//
// Default constructor/destructor/move
StackAllocatorContainer::StackAllocatorContainer()  = default;
StackAllocatorContainer::~StackAllocatorContainer() = default;
StackAllocatorContainer::StackAllocatorContainer(StackAllocatorContainer&&)
    = default;
StackAllocatorContainer&
StackAllocatorContainer::operator=(StackAllocatorContainer&&)
    = default;

//---------------------------------------------------------------------------//
/*!
 * Get a view to the managed data.
 */
StackAllocatorView StackAllocatorContainer::device_view() const
{
    REQUIRE(data_);
    StackAllocatorView view;
    view.data     = thrust::raw_pointer_cast(data_->storage.data());
    view.size     = reinterpret_cast<size_type*>(view.data + capacity_);
    view.capacity = capacity_;
    return view;
}

//---------------------------------------------------------------------------//
/*!
 * Clear allocated data.
 */
void StackAllocatorContainer::clear()
{
    celeritas::KernelParamCalculator calc_launch_params;
    auto                             params = calc_launch_params(1);
    stack_allocator_clear_impl<<<params.grid_size, params.block_size>>>(
        this->device_view());
    CELER_CUDA_CHECK_ERROR();
}

//---------------------------------------------------------------------------//
/*!
 * Swap with another stack allocator.
 */
void StackAllocatorContainer::swap(StackAllocatorContainer& other)
{
    std::swap(*this, other);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
