//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Particle.test.hh
//---------------------------------------------------------------------------//

#include <vector>
#include "physics/base/ParticleInterface.hh"

namespace celeritas_test
{
using namespace celeritas;

using ParticleParamsPointers
    = ParticleParamsData<Ownership::const_reference, MemSpace::device>;
using ParticleStatePointers
    = ParticleStateData<Ownership::reference, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct PTVTestInput
{
    ParticleParamsPointers          params;
    ParticleStatePointers           states;
    std::vector<ParticleTrackState> init;
};

//---------------------------------------------------------------------------//
//! Output results
struct PTVTestOutput
{
    std::vector<double> props;

    static CELER_CONSTEXPR_FUNCTION int props_per_thread() { return 8; }
};

//---------------------------------------------------------------------------//
//! Run on device and return results
PTVTestOutput ptv_test(PTVTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
