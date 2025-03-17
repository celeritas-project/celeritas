//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/StatusChecker.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/track/StatusChecker.hh"

#include "corecel/Types.hh"
#include "corecel/data/AuxParamsRegistry.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/ActionRegistry.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreState.hh"
#include "celeritas/track/SimTrackView.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
template<class T, Ownership W, Ownership W2, class I>
void copy_to_device(Collection<T, W, MemSpace::host, I> const& src,
                    Collection<T, W2, MemSpace::device, I> const& dst)
{
    CELER_EXPECT(src.size() == dst.size());
    Copier<T, MemSpace::device> copy_to_device{
        dst[AllItems<T, MemSpace::device>{}]};
    copy_to_device(MemSpace::host, src[AllItems<T, MemSpace::host>{}]);
}

//---------------------------------------------------------------------------//

class StatusCheckerTest : public SimpleTestBase
{
  protected:
    void SetUp() override
    {
        // Disable base-class status checker
        this->disable_status_checker();

        // Create status checker
        auto& action_reg = *this->action_reg();
        auto& aux_reg = *this->aux_reg();
        status_checker_ = std::make_shared<StatusChecker>(action_reg.next_id(),
                                                          aux_reg.next_id());
        aux_reg.insert(status_checker_);
        action_reg.insert(status_checker_);
    }

    // Create primary particles
    std::vector<Primary> make_primaries(size_type num_primaries) const
    {
        std::vector<Primary> result;
        for (unsigned int i = 0; i < num_primaries; ++i)
        {
            Primary p;
            p.particle_id = ParticleId{0};
            p.energy = units::MevEnergy(1 + i);
            p.position = {0, 0, 0};
            p.direction = {0, 0, 1};
            p.time = 0;
            p.event_id = EventId{0};
            result.push_back(p);
        }
        return result;
    }

    ActionId find_action(std::string const& label) const
    {
        auto& reg = *this->action_reg();
        auto id = reg.find_action(label);
        CELER_VALIDATE(id, << "no action '" << label << '\'');
        return id;
    }

    std::string const& id_to_label(ActionId id) const
    {
        return this->action_reg()->id_to_label(id);
    }

    template<MemSpace M>
    void begin_run(CoreState<M>& state)
    {
        status_checker_->begin_run(*this->core(), state);
    }

    template<MemSpace M>
    void run(ActionId id, CoreState<M>& state)
    {
        CELER_EXPECT(id);

        // Execute the action
        dynamic_cast<CoreStepActionInterface const&>(
            *this->action_reg()->action(id))
            .step(*this->core(), state);

        // Run the status checker
        status_checker_->step(id, *this->core(), state);
    }

    template<MemSpace M>
    void
    check_throw(ActionId id, CoreState<M>& state, std::string_view match) const
    {
        CELER_EXPECT(id);

        // Run the status checker
        std::string actual_message;
        try
        {
            status_checker_->step(id, *this->core(), state);
        }
        catch (RichContextException const& e)
        {
            CELER_LOG(info) << "Exception during action " << id.get()
                            << " came from: " << e.what();
            actual_message = "<Failed to rethrow rich context correctly>";
            try
            {
                // Rethrow to print the underlying message
                std::rethrow_if_nested(e);
            }
            catch (DebugError const& e)
            {
                actual_message = e.details().condition;
            }
        }
        catch (DebugError const& e)
        {
            actual_message = e.details().condition;
        }
        catch (RuntimeError const& e)
        {
            // This gets thrown if a CUDA assert is triggered
            actual_message = e.details().what;
        }

        EXPECT_TRUE(actual_message.find(match) != std::string::npos)
            << "Actual message: '" << actual_message << "'";
    }

  protected:
    std::shared_ptr<StatusChecker> status_checker_;
};

TEST_F(StatusCheckerTest, host)
{
    CoreState<MemSpace::host> state{*this->core(), StreamId{0}, 16};
    this->begin_run(state);
    this->insert_primaries(state, make_span(this->make_primaries(8)));

    // Keep a persistent view to the last track slot
    SimTrackView sim(
        core()->ref<MemSpace::host>().sim, state.ref().sim, TrackSlotId{15});

    // Actions: see Stepper.test.cc
    EXPECT_EQ(TrackStatus::inactive, sim.status());
    auto id = this->find_action("extend-from-primaries");
    this->run(id, state);
    EXPECT_EQ(TrackStatus::inactive, sim.status());

    id = this->find_action("initialize-tracks");
    this->run(id, state);
    EXPECT_EQ(TrackStatus::initializing, sim.status());

    // Run pre-step, set the state to something bad, check that it throws
    id = this->find_action("pre-step");
    this->run(id, state);
    EXPECT_EQ(TrackStatus::alive, sim.status());
    sim.status(TrackStatus::initializing);
    this->check_throw(id, state, "status was improperly reverted");
    // ...and restore the state
    sim.status(TrackStatus::alive);

    // Run kernel to select a model
    id = this->find_action("physics-discrete-select");
    this->run(id, state);

    // Set a bogus along-step state
    sim.along_step_action(this->find_action("pre-step"));
    this->check_throw(id, state, "along-step action cannot yet change");
    sim.along_step_action(this->find_action("along-step-neutral"));

    // Run klein-nishina, change the ID *back* to an earlier action
    id = this->find_action("scat-klein-nishina");
    this->run(id, state);
    EXPECT_EQ("scat-klein-nishina", this->id_to_label(sim.post_step_action()));
    sim.post_step_action(this->find_action("physics-discrete-select"));
    this->check_throw(id, state, "new post-step action is out of order");
}

TEST_F(StatusCheckerTest, TEST_IF_CELER_DEVICE(device))
{
    device().create_streams(1);
    CoreState<MemSpace::device> state{*this->core(), StreamId{0}, 128};
    this->begin_run(state);
    this->insert_primaries(state, make_span(this->make_primaries(64)));

    // Check that the first half of a stepping loop is fine
    for (auto label : {"extend-from-primaries",
                       "initialize-tracks",
                       "pre-step",
                       "physics-discrete-select",
                       "scat-klein-nishina"})
    {
        auto id = this->find_action(label);
        this->run(id, state);
    }

    // Incorrectly and hackily adjust the state
    TrackSlotId const target_track{72};
    StateCollection<ActionId, Ownership::value, MemSpace::host> post_step;
    post_step = state.ref().sim.post_step_action;
    ASSERT_EQ(post_step.size(), state.size());
    ASSERT_NE(ActionId{}, post_step[target_track]);
    EXPECT_EQ("scat-klein-nishina", this->id_to_label(post_step[target_track]));
    post_step[target_track] = this->find_action("physics-discrete-select");
    copy_to_device(post_step, state.ref().sim.post_step_action);

    if constexpr (CELERITAS_USE_HIP)
    {
        GTEST_SKIP() << "HIP debug calls 'abort' rather than asserting";
    }

    EXPECT_THROW(status_checker_->step(this->find_action("scat-klein-nishina"),
                                       *this->core(),
                                       state),
                 RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
