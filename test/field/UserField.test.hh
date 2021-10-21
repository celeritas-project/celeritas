//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file UserField.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>
#include "base/Types.hh"
#include "detail/FieldMapData.hh"

namespace celeritas_test
{
using namespace celeritas;

using celeritas::MemSpace;
using celeritas::Ownership;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

//! Test parameters
struct UserFieldTestParams
{
    //  using size_type = celeritas::size_type;
    size_type nsamples; //! number of sampling points
    real_type delta_z;  //! delta for the z-position
    real_type delta_r;  //! delta for the r-position
};

//! Output results
struct UserFieldTestOutput
{
    using real_type = celeritas::real_type;
    std::vector<real_type> value_x;
    std::vector<real_type> value_y;
    std::vector<real_type> value_z;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
UserFieldTestOutput fieldmap_test(UserFieldTestParams test_param,
                                  celeritas::detail::FieldMapDeviceRef group);

#if !CELERITAS_USE_CUDA
inline UserFieldTestOutput
fieldmap_test(UserFieldTestParams,
              CELER_MAYBE_UNUSED celeritas::detail::FieldMapDeviceRef group)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
//! Run on device and return results
UserFieldTestOutput parameterized_field_test(UserFieldTestParams test_param);

#if !CELERITAS_USE_CUDA
inline UserFieldTestOutput parameterized_field_test(UserFieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
