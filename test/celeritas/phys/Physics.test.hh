//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/Physics.test.hh
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/grid/XsCalculator.hh"
#include "celeritas/phys/PhysicsData.hh"

// Kernel functions
#include "corecel/math/Algorithms.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/phys/PhysicsStepView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct PhysTestInit
{
    units::MevEnergy energy;
    PhysMatId mat;
    ParticleId particle;
};

struct PTestInput
{
    DeviceCRef<PhysicsParamsData> params;
    DeviceRef<PhysicsStateData> states;
    DeviceCRef<ParticleParamsData> par_params;
    DeviceRef<ParticleStateData> par_states;
    DeviceCRef<MaterialParamsData> mat_params;
    StateCollection<PhysTestInit, Ownership::const_reference, MemSpace::device>
        inits;

    // Calculated "step" per track
    Span<real_type> result;
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
inline CELER_FUNCTION real_type calc_step(PhysicsTrackView& phys,
                                          PhysicsStepView& pstep,
                                          MaterialView const& mat,
                                          units::MevEnergy energy)
{
    // Calc total macro_xs over processes
    real_type total_xs = 0;
    for (auto ppid : range(ParticleProcessId{phys.num_particle_processes()}))
    {
        real_type process_xs = 0;
        if (auto id = phys.macro_xs_grid(ppid))
        {
            process_xs = phys.calc_xs(ppid, mat, energy);
        }

        // Zero cross section if outside of model range
        auto find_model = phys.make_model_finder(ppid);
        if (!find_model(energy))
        {
            process_xs = 0;
        }

        pstep.per_process_xs(ppid) = process_xs;
        total_xs += process_xs;
    }
    phys.interaction_mfp(1 / total_xs);

    // Calc minimum range
    auto const inf = numeric_limits<real_type>::infinity();
    real_type step = inf;
    if (auto id = phys.range_grid())
    {
        auto calc_range = phys.make_calculator<RangeCalculator>(id);
        step = min(step, calc_range(energy));
    }
    if (step != inf)
    {
        step = phys.range_to_step(step);
    }

    // Take minimum of step and half the MFP
    step = min(step, real_type{0.5} * phys.interaction_mfp());
    return step;
}

//---------------------------------------------------------------------------//
//! Run on device and return results
void phys_cuda_test(PTestInput const&);

#if !CELER_USE_DEVICE
inline void phys_cuda_test(PTestInput const&)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
