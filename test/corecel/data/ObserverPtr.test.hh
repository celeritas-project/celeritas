//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/ObserverPtr.test.hh
//---------------------------------------------------------------------------//

#pragma once

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/ObserverPtr.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// DEVICE KERNEL EXECUTION
//---------------------------------------------------------------------------//
void copy_test(ObserverPtr<int const, MemSpace::device> in_data,
               ObserverPtr<int, MemSpace::device> out_data,
               size_type size);
void copy_thrust_test(ObserverPtr<int const, MemSpace::device> in_data,
                      ObserverPtr<int, MemSpace::device> out_data,
                      size_type size);

//---------------------------------------------------------------------------//
#if !CELER_USE_DEVICE
inline void copy_test(ObserverPtr<int const, MemSpace::device>,
                      ObserverPtr<int, MemSpace::device>,
                      size_type)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline void copy_thrust_test(ObserverPtr<int const, MemSpace::device>,
                             ObserverPtr<int, MemSpace::device>,
                             size_type)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
