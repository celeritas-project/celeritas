//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalCollector.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/OpticalCollector.hh"

#include <memory>
#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/io/LogContextException.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/global/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/optical/OpticalCollector.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "celeritas_test.hh"
#include "../LArSphereBase.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TEST FIXTURES
//---------------------------------------------------------------------------//

class LArSphereCollectorTest : public LArSphereBase
{
  public:
    void SetUp() override
    {
        size_type stack_capacity = 128;
        size_type num_streams = 1;
        auto& action_reg = *this->action_reg();
        collector_ = std::make_shared<OpticalCollector>(this->properties(),
                                                        this->cerenkov(),
                                                        this->scintillation(),
                                                        stack_capacity,
                                                        num_streams,
                                                        &action_reg);
    }

    SPConstAction build_along_step() override
    {
        auto& action_reg = *this->action_reg();
        UniformFieldParams field_params;
        field_params.field = {0, 0, 1 * units::tesla};
        auto msc = UrbanMscParams::from_import(
            *this->particle(), *this->material(), this->imported_data());

        auto result = std::make_shared<AlongStepUniformMscAction>(
            action_reg.next_id(), field_params, nullptr, msc);
        CELER_ASSERT(result);
        CELER_ASSERT(result->has_msc());
        action_reg.insert(result);
        return result;
    }

    std::vector<Primary> make_primaries(size_type count)
    {
        Primary p;
        p.event_id = EventId{0};
        p.energy = MevEnergy{10.0};
        p.position = from_cm(Real3{0, 0, 0});
        p.direction = Real3{1, 0, 0};
        p.time = 0;
        p.particle_id = this->particle()->find(pdg::electron());
        CELER_ASSERT(p.particle_id);
        std::vector<Primary> result(count, p);

        for (auto i : range(count))
        {
            result[i].track_id = TrackId{i};
        }
        return result;
    }

    template<MemSpace M>
    void run(size_type num_tracks, size_type num_steps)
    {
        StepperInput step_inp;
        step_inp.params = this->core();
        step_inp.stream_id = StreamId{0};
        step_inp.num_track_slots = num_tracks;

        Stepper<M> step(step_inp);
        LogContextException log_context{this->output_reg().get()};

        // Initial step
        auto primaries = this->make_primaries(num_tracks);
        StepperResult count;
        CELER_TRY_HANDLE(count = step(make_span(primaries)), log_context);

        while (count && --num_steps > 0)
        {
            CELER_TRY_HANDLE(count = step(), log_context);
        }
    }

  private:
    using SPOpticalCollector = std::shared_ptr<OpticalCollector const>;

    SPOpticalCollector collector_;
};

template void LArSphereCollectorTest::run<MemSpace::host>(size_type, size_type);
template void
    LArSphereCollectorTest::run<MemSpace::device>(size_type, size_type);

//---------------------------------------------------------------------------//
// TESTEM3
//---------------------------------------------------------------------------//

TEST_F(LArSphereCollectorTest, host)
{
    // TODO: Check
    this->run<MemSpace::host>(8, 8);
}

TEST_F(LArSphereCollectorTest, TEST_IF_CELER_DEVICE(device))
{
    // TODO: Check
    this->run<MemSpace::device>(8, 8);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
