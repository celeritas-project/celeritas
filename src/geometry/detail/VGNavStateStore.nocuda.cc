//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStateStore.nocuda.cc
//---------------------------------------------------------------------------//
#include "VGNavStateStore.hh"

#include "base/Assert.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Prevent allocation because CUDA is disabled.
 */
VGNavStateStore::VGNavStateStore(size_type, int)
{
    throw DebugError("Cannot allocate device memory because CUDA is disabled");
}

//---------------------------------------------------------------------------//
/*!
 * Device view cannot be called when CUDA is disabled.
 */
void* VGNavStateStore::device_pointers() const
{
    REQUIRE(false);
    return nullptr;
}

//---------------------------------------------------------------------------//
//! Deleter should never be called on CPU
void VGNavStateStore::NavStatePoolDeleter::operator()(NavStatePool* ptr) const
{
    REQUIRE(false);
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas
