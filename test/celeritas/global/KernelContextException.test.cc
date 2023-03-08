//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/KernelContextException.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/global/KernelContextException.hh"

#include <exception>

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/io/JsonPimpl.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "../SimpleTestBase.hh"
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
#if CELERITAS_USE_JSON
    JsonPimpl jp;
    e.output(&jp);
    return jp.obj.dump();
#else
    (void)sizeof(e);
    return {};
#endif
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
        p.position = {0, 1, 0};
        p.direction = {0, 0, 1};
        p.time = 0;

        CELER_ASSERT(p.particle_id);

        std::vector<Primary> result(count, p);

        for (auto i : range(count))
        {
            result[i].event_id = EventId{i / (count / 2)};
            result[i].track_id = TrackId{i % (count / 2)};
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
    this->check_kce = [](KernelContextException const& e) {
        EXPECT_STREQ(
            "kernel context: thread 15 in 'test-kernel', track 3 of event 1",
            e.what());
        EXPECT_EQ(TrackSlotId{15}, e.thread());
        EXPECT_EQ(EventId{1}, e.event());
        EXPECT_EQ(TrackId{3}, e.track());
        EXPECT_EQ(TrackId{}, e.parent());
        EXPECT_EQ(1, e.num_steps());
        EXPECT_EQ(ParticleId{0}, e.particle());
        EXPECT_EQ(10, e.energy().value());
        EXPECT_VEC_SOFT_EQ((Real3{0, 1, 5}), e.pos());
        EXPECT_VEC_SOFT_EQ((Real3{0, 0, 1}), e.dir());
        if (!CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ(VolumeId{2}, e.volume());
            EXPECT_EQ(SurfaceId{11}, e.surface());
        }
        EXPECT_EQ(SurfaceId{}, e.next_surface());
        if (CELERITAS_USE_JSON && !CELERITAS_USE_VECGEOM)
        {
            EXPECT_EQ(
                R"json({"dir":[0.0,0.0,1.0],"energy":[10.0,"MeV"],"event":1,"label":"test-kernel","num_steps":1,"particle":0,"pos":[0.0,1.0,5.0],"surface":11,"thread":15,"track":3,"volume":2})json",
                get_json_str(e));
        }
    };
    // Since tracks are initialized back to front, the thread ID must be toward
    // the end
    CELER_TRY_HANDLE_CONTEXT(
        throw DebugError({DebugErrorType::internal, "false", "test.cc", 0}),
        this->check_exception,
        KernelContextException(
            step.core_data(), TrackSlotId{15}, "test-kernel"));
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
        EXPECT_STREQ("kernel context: thread 1 in 'test-kernel'", e.what());
        EXPECT_EQ(TrackSlotId{1}, e.thread());
        EXPECT_EQ(EventId{}, e.event());
        EXPECT_EQ(TrackId{}, e.track());
        if (CELERITAS_USE_JSON)
        {
            EXPECT_EQ(R"json({"label":"test-kernel","thread":1})json",
                      get_json_str(e));
        }
    };
    CELER_TRY_HANDLE_CONTEXT(
        throw DebugError({DebugErrorType::internal, "false", "test.cc", 0}),
        this->check_exception,
        KernelContextException(
            step.core_data(), TrackSlotId{1}, "test-kernel"));
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
        EXPECT_EQ(TrackSlotId{}, e.thread());
        if (CELERITAS_USE_JSON)
        {
            EXPECT_EQ(R"json({"label":"dumb-kernel"})json", get_json_str(e));
        }
    };
    CELER_TRY_HANDLE_CONTEXT(
        throw DebugError({DebugErrorType::internal, "false", "test.cc", 0}),
        this->check_exception,
        KernelContextException(step.core_data(), TrackSlotId{}, "dumb-kernel"));
    EXPECT_TRUE(this->caught_debug);
    EXPECT_TRUE(this->caught_kce);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
