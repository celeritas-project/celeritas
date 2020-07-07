//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file StackAllocatorView.hh
//---------------------------------------------------------------------------//
#ifndef base_StackAllocatorView_hh
#define base_StackAllocatorView_hh

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Reference data owned by a StackAllocatorContainer for use in StackAllocator.
 */
struct StackAllocatorView
{
    //! Size type needed for CUDA compatibility
    using size_type = unsigned long long int;

    char*      data;
    size_type* size;
    size_type  capacity;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#endif // base_StackAllocatorView_hh
