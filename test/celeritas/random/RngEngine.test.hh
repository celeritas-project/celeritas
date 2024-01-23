//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/RngEngine.test.hh
//---------------------------------------------------------------------------//

#include <vector>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/random/RngData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
using RngDeviceParamsRef = DeviceCRef<RngParamsData>;
using RngDeviceStateRef = DeviceRef<RngStateData>;

//---------------------------------------------------------------------------//
//! Run on device and return results
std::vector<unsigned int> re_test_native(RngDeviceParamsRef, RngDeviceStateRef);

template<class T>
std::vector<T> re_test_canonical(RngDeviceParamsRef, RngDeviceStateRef);

#if !CELER_USE_DEVICE
std::vector<unsigned int> re_test_native(RngDeviceParamsRef, RngDeviceStateRef)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}

template<class T>
inline std::vector<T> re_test_canonical(RngDeviceParamsRef, RngDeviceStateRef)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
