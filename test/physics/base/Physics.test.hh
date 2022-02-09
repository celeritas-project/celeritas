//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Physics.test.hh
//---------------------------------------------------------------------------//
#include "celeritas_config.h"
#include "base/Assert.hh"
#include "base/Collection.hh"
#include "base/Range.hh"
#include "base/Span.hh"
#include "physics/base/PhysicsData.hh"
#include "physics/base/Units.hh"
#include "physics/base/Types.hh"
#include "physics/grid/RangeCalculator.hh"
#include "physics/grid/XsCalculator.hh"
#include "physics/material/Types.hh"

// Kernel functions
#include "base/Algorithms.hh"
#include "base/NumericLimits.hh"
#include "physics/base/PhysicsTrackView.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct PhysTestInit
{
    celeritas::units::MevEnergy energy;
    celeritas::MaterialId       mat;
    celeritas::ParticleId       particle;
};

struct PTestInput
{
    celeritas::PhysicsParamsData<Ownership::const_reference, MemSpace::device>
                                                                        params;
    celeritas::PhysicsStateData<Ownership::reference, MemSpace::device> states;
    celeritas::StateCollection<PhysTestInit,
                               Ownership::const_reference,
                               MemSpace::device>
        inits;

    // Calculated "step" per track
    celeritas::Span<celeritas::real_type> result;
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
inline CELER_FUNCTION celeritas::real_type
                      calc_step(celeritas::PhysicsTrackView& phys, celeritas::units::MevEnergy energy)
{
    using namespace celeritas;

    // Calc total macro_xs over processsess
    real_type total_xs = 0;
    for (auto ppid : range(ParticleProcessId{phys.num_particle_processes()}))
    {
        real_type process_xs = 0;
        if (auto id = phys.value_grid(ValueGridType::macro_xs, ppid))
        {
            auto calc_xs = phys.make_calculator<XsCalculator>(id);
            process_xs   = calc_xs(energy);
        }

        // Zero cross section if outside of model range
        auto find_model = phys.make_model_finder(ppid);
        if (!find_model(energy))
        {
            process_xs = 0;
        }

        phys.per_process_xs(ppid) = process_xs;
        total_xs += process_xs;
    }
    phys.interaction_mfp(1 / total_xs);

    // Calc minimum range
    const auto inf  = numeric_limits<real_type>::infinity();
    real_type  step = inf;
    for (auto ppid : range(ParticleProcessId{phys.num_particle_processes()}))
    {
        if (auto id = phys.value_grid(ValueGridType::range, ppid))
        {
            auto calc_range = phys.make_calculator<RangeCalculator>(id);
            step            = celeritas::min(step, calc_range(energy));
        }
    }
    if (step != inf)
    {
        step = phys.range_to_step(step);
    }

    // Take minimum of step and half the MFP
    step = celeritas::min(step, 0.5 * phys.interaction_mfp());
    return step;
}

//---------------------------------------------------------------------------//
//! Run on device and return results
void phys_cuda_test(const PTestInput&);

#if !CELERITAS_USE_CUDA
inline void phys_cuda_test(const PTestInput&)
{
    CELER_NOT_CONFIGURED("CUDA");
}
#endif

//---------------------------------------------------------------------------//
} // namespace celeritas_test
