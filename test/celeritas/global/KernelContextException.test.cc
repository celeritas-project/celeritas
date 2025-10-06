//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/KernelContextException.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/KernelContextException.hh"

#include <exception>
#include <regex>
#include <sstream>
#include <string>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/StringSimplifier.hh"
#include "corecel/io/JsonPimpl.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "StepperTestBase.hh"
#include "celeritas_test.hh"

using celeritas::units::MevEnergy;

namespace celeritas
{
namespace test
{
namespace
{
std::string get_json_str(KernelContextException const& e)
{
    JsonPimpl jp;
    e.output(&jp);
    return jp.obj.dump();
}

ThreadId find_thread(HostRef<CoreStateData> const& state, TrackSlotId track)
{
    CELER_EXPECT(state.track_slots.empty());
    return ThreadId{track.get()};
}

}  // namespace

//---------------------------------------------------------------------------//

class KernelContextExceptionTest : public SimpleTestBase, public StepperTestBase
{
  protected:
    std::vector<Primary> make_primaries(size_type count) const override
    {
        Primary p;
        p.particle_id = this->particle()->find(pdg::gamma());
        p.energy = MevEnergy{10};
        p.position = from_cm(Real3{0, 1, 0});
        p.direction = {0, 0, 1};
        p.time = 0;

        CELER_ASSERT(p.particle_id);

        std::vector<Primary> result(count, p);

        for (auto i : range(count))
        {
            result[i].event_id = EventId{i / (count / 2)};
        }
        return result;
    }

    size_type max_average_steps() const override { return 1000; }

    void check_exception(std::exception_ptr eptr)
    {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch (KernelContextException const& e)
        {
            caught_kce = true;
            if (this->check_kce)
            {
                this->check_kce(e);
            }
            try
            {
                std::rethrow_if_nested(e);
            }
            catch (DebugError const& e)
            {
                caught_debug = true;
                EXPECT_STREQ("test.cc", e.details().file);
            }
        }
    }

    std::function<void(KernelContextException const&)> check_kce;
    bool caught_debug = false;
    bool caught_kce = false;
};

TEST_F(KernelContextExceptionTest, typical)
{
    // Create some track slots
    Stepper<MemSpace::host> step(this->make_stepper_input(16));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    // Check for these values based on the step count and thread ID below
    this->check_kce = [&step](KernelContextException const& e) {
        auto simplified_str = StringSimplifier{3}(e.what());
        // Remove labels for reperoducibility
        simplified_str = std::regex_replace(
            simplified_str, std::regex("@(global|world)"), "");

        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS)
        {
            EXPECT_EQ(
                R"(track slot 15 in kernel 'test-kernel': {"geo":{"dir":[0.0,0.0,1.0],"is_on_boundary":true,"is_outside":false,"pos":[[0.0,1.0,5.0],"cm"],"volume_id":"world"},"mat":"hard vacuum","particle":{"energy":[10.0,"MeV"],"particle_id":"gamma"},"sim":{"along_step_action":"along-step-neutral","event_id":1,"num_steps":1,"parent_id":null,"post_step_action":"geo-boundary","status":"alive","step_length":[5.0,"cm"],"time":[1.67e-10,"s"],"track_id":3},"thread_id":15,"track_slot_id":15})",
                simplified_str)
                << repr(simplified_str);
        }

        EXPECT_EQ(find_thread(step.state_ref(), TrackSlotId{15}), e.thread());
        EXPECT_EQ(TrackSlotId{15}, e.track_slot());
        EXPECT_EQ(EventId{1}, e.event());
        EXPECT_EQ(TrackId{3}, e.track());
        EXPECT_EQ(TrackId{}, e.parent());
        EXPECT_EQ(1, e.num_steps());
        EXPECT_EQ(ParticleId{0}, e.particle());
        EXPECT_EQ(10, e.energy().value());
        EXPECT_VEC_SOFT_EQ(from_cm(Real3{0, 1, 5}), e.pos());
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1}), e.dir());
        if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            EXPECT_EQ(ImplVolumeId{2}, e.volume());
            EXPECT_EQ(ImplSurfaceId{11}, e.surface());
        }
        if (CELERITAS_UNITS == CELERITAS_UNITS_CGS
            && CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            std::stringstream ss;
            ss << R"json({"dir":[0.0,0.0,1.0],"energy":[10.0,"MeV"],"event":1,"label":"test-kernel","num_steps":1,"particle":0,"pos":[0.0,1.0,5.0],"surface":11,"thread":)json"
               << e.thread().unchecked_get()
               << R"json(,"track":3,"track_slot":15,"volume":2})json";
            EXPECT_JSON_EQ(ss.str(), get_json_str(e));
        }
    };
    // Since tracks are initialized back to front, the thread ID must be toward
    // the end
    CELER_TRY_HANDLE_CONTEXT(
        throw DebugError({DebugErrorType::internal, "false", "test.cc", 0}),
        this->check_exception,
        KernelContextException(this->core()->ref<MemSpace::host>(),
                               step.state_ref(),
                               find_thread(step.state_ref(), TrackSlotId{15}),
                               "test-kernel"));
    EXPECT_TRUE(this->caught_debug);
    EXPECT_TRUE(this->caught_kce);
}

TEST_F(KernelContextExceptionTest, uninitialized_track)
{
    // Create some track slots
    Stepper<MemSpace::host> step(this->make_stepper_input(8));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(4);
    step(make_span(primaries));

    this->check_kce = [](KernelContextException const& e) {
        // Don't test this with vecgeom which has more assertions when
        // acquiring data
        EXPECT_STREQ("track slot 1 in kernel 'test-kernel'", e.what());
        EXPECT_EQ(TrackSlotId{1}, e.track_slot());
        EXPECT_EQ(EventId{}, e.event());
        EXPECT_EQ(TrackId{}, e.track());

        {
            std::stringstream ss;
            ss << R"json({"label":"test-kernel","thread":)json"
               << e.thread().unchecked_get() << R"json(,"track_slot":1})json";
            EXPECT_JSON_EQ(ss.str(), get_json_str(e));
        }
    };

    CELER_TRY_HANDLE_CONTEXT(
        throw DebugError({DebugErrorType::internal, "false", "test.cc", 0}),
        this->check_exception,
        KernelContextException(this->core()->ref<MemSpace::host>(),
                               step.state_ref(),
                               find_thread(step.state_ref(), TrackSlotId{1}),
                               "test-kernel"));
    EXPECT_TRUE(this->caught_debug);
    EXPECT_TRUE(this->caught_kce);
}

TEST_F(KernelContextExceptionTest, bad_thread)
{
    // Create some track slots
    Stepper<MemSpace::host> step(this->make_stepper_input(4));

    // Initialize some primaries and take a step
    auto primaries = this->make_primaries(8);
    step(make_span(primaries));

    this->check_kce = [](KernelContextException const& e) {
        EXPECT_STREQ("dumb-kernel (error processing track state)", e.what());
        EXPECT_EQ(TrackSlotId{}, e.track_slot());
        EXPECT_JSON_EQ(R"json({"label":"dumb-kernel"})json", get_json_str(e));
    };
    CELER_TRY_HANDLE_CONTEXT(
        throw DebugError({DebugErrorType::internal, "false", "test.cc", 0}),
        this->check_exception,
        KernelContextException(this->core()->ref<MemSpace::host>(),
                               step.state_ref(),
                               ThreadId{},
                               "dumb-kernel"));
    EXPECT_TRUE(this->caught_debug);
    EXPECT_TRUE(this->caught_kce);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
