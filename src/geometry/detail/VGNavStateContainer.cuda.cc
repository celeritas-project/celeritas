//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStateContainer.cuda.cc
//---------------------------------------------------------------------------//
#include "VGNavStateContainer.hh"

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
VGNavStateContainer::VGNavStateContainer(size_type size, int depth)
{
    pool_ = std::make_unique<vecgeom::cxx::NavStatePool>(size, depth);
    pool_->CopyToGpu();
}

//---------------------------------------------------------------------------//
// Default
VGNavStateContainer::VGNavStateContainer()                      = default;
VGNavStateContainer::~VGNavStateContainer()                     = default;
VGNavStateContainer::VGNavStateContainer(VGNavStateContainer&&) = default;
VGNavStateContainer& VGNavStateContainer::operator=(VGNavStateContainer&&)
    = default;

//---------------------------------------------------------------------------//
/*!
 * Get allocated GPU state pointer
 */
void* VGNavStateContainer::device_view() const
{
    REQUIRE(pool_);
    void* ptr = pool_->GetGPUPointer();
    ENSURE(ptr);
    return ptr;
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
