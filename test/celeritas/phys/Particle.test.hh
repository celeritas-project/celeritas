//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Particle.test.hh
//---------------------------------------------------------------------------//

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "celeritas/phys/ParticleData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
//! Input data
struct PTVTestInput
{
    DeviceCRef<ParticleParamsData> params;
    DeviceRef<ParticleStateData> states;

    std::vector<ParticleTrackInitializer> init;
};

//---------------------------------------------------------------------------//
//! Output results
struct PTVTestOutput
{
    std::vector<real_type> props;

    static CELER_CONSTEXPR_FUNCTION int props_per_thread() { return 8; }
};

//---------------------------------------------------------------------------//
//! Run on device and return results
PTVTestOutput ptv_test(PTVTestInput);

#if !CELER_USE_DEVICE
inline PTVTestOutput ptv_test(PTVTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
