//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file field/UserField.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"

#include "FieldPropagator.test.hh"
#include "detail/FieldMapData.hh"

namespace celeritas_test
{
using namespace celeritas;

using celeritas::MemSpace;
using celeritas::Ownership;
using celeritas::detail::FieldMapDeviceRef;

using UserFieldTestVector = std::vector<double>;

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
UserFieldTestOutput parameterized_field_test(UserFieldTestParams test_param);

UserFieldTestVector par_fp_test(FPTestInput input);

UserFieldTestVector par_bc_test(FPTestInput input);

UserFieldTestOutput
fieldmap_test(UserFieldTestParams test_param, FieldMapDeviceRef data);

UserFieldTestVector map_fp_test(FPTestInput input, FieldMapDeviceRef data);

UserFieldTestVector map_bc_test(FPTestInput input, FieldMapDeviceRef data);

#if !CELER_USE_DEVICE
inline UserFieldTestOutput parameterized_field_test(UserFieldTestParams)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline UserFieldTestVector par_fp_test(FPTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline UserFieldTestVector par_bc_test(FPTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline UserFieldTestOutput
fieldmap_test(UserFieldTestParams, CELER_MAYBE_UNUSED FieldMapDeviceRef data)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline UserFieldTestVector
map_fp_test(FPTestInput, CELER_MAYBE_UNUSED FieldMapDeviceRef data)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
inline UserFieldTestVector
map_bc_test(FPTestInput, CELER_MAYBE_UNUSED FieldMapDeviceRef data)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
