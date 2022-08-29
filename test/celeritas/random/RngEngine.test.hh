//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
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
using RngDeviceRef = celeritas::DeviceRef<celeritas::RngStateData>;

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
//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
