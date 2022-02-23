//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RngEngine.test.hh
//---------------------------------------------------------------------------//

#include <vector>

#include "base/Assert.hh"
#include "base/Macros.hh"
#include "base/Types.hh"
#include "random/RngData.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
using RngDeviceRef
    = celeritas::RngStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
//! Run on device and return results
std::vector<unsigned int> re_test_native(RngDeviceRef);

template<class T>
std::vector<T> re_test_canonical(RngDeviceRef);

#if !CELER_USE_DEVICE
std::vector<unsigned int> re_test_native(RngDeviceRef)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<class T>
inline std::vector<T> re_test_canonical(RngDeviceRef)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
