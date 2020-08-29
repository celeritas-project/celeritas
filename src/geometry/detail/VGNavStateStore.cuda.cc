//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStateStore.cuda.cc
//---------------------------------------------------------------------------//
#include "VGNavStateStore.hh"

#include <VecGeom/navigation/NavStatePool.h>
#include "base/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with sizes, allocating on GPU.
 */
VGNavStateStore::VGNavStateStore(size_type size, int depth)
{
    pool_.reset(new vecgeom::cxx::NavStatePool(size, depth));
}

//---------------------------------------------------------------------------//
/*!
 * Copy host states to device.
 */
void VGNavStateStore::copy_to_device()
{
    REQUIRE(*this);
    pool_->CopyToGpu();
    CELER_CUDA_CALL(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------//
/*!
 * Get allocated GPU state pointer.
 *
 * copy_to_device must be called before this. If not, the GPU pointer may be
 * null.
 */
void* VGNavStateStore::device_pointers() const
{
    REQUIRE(*this);
    void* ptr = pool_->GetGPUPointer();
    ENSURE(ptr);
    return ptr;
}

//---------------------------------------------------------------------------//
//! Deleter frees cuda data
void VGNavStateStore::NavStatePoolDeleter::operator()(NavStatePool* ptr) const
{
    delete ptr;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
