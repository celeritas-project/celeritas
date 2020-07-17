//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Particle.test.hh
//---------------------------------------------------------------------------//

#include <vector>
#include "physics/base/ParticleParamsPointers.hh"
#include "physics/base/ParticleStatePointers.hh"

namespace celeritas_test
{
using namespace celeritas;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct TPVTestInput
{
    ParticleParamsPointers          params;
    ParticleStatePointers           states;
    std::vector<ParticleTrackState> init;
};

//---------------------------------------------------------------------------//
//! Output results
struct TPVTestOutput
{
    std::vector<double> props;

    static CELER_CONSTEXPR_FUNCTION int props_per_thread() { return 8; }
};

//---------------------------------------------------------------------------//
//! Run on device and return results
TPVTestOutput tpv_test(TPVTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
