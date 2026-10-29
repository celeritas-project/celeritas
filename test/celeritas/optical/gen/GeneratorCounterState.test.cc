//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorCounterState.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/gen/GeneratorCounterState.hh"

#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/optical/gen/detail/GeneratorCounterAlgorithms.hh"
#include "celeritas/track/CounterAlgorithms.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
//---------------------------------------------------------------------------//

class GeneratorCounterStateTest : public ::celeritas::test::Test
{
  protected:
    using CountersT = GeneratorCounters<>;
    using HostPtr = ObserverPtr<size_type const, MemSpace::host>;

    void SetUp() override
    {
        resize(&core_, 1);
        core_ref_ = CoreStateCounterRef<MemSpace::host>{core_};
    }

    //! Set the core state counters
    void set_core(size_type num_vacancies, size_type num_pending)
    {
        auto& core = this->core();
        core = {};
        core.num_vacancies = num_vacancies;
        core.num_pending = num_pending;
    }

    //! Access the core state counters
    CoreStateCounters& core() { return core_[ItemId<CoreStateCounters>{0}]; }

    //! Generate photons into the vacancies, as the generator actions do
    void update(GeneratorCounterState<MemSpace::host>& state)
    {
        detail::update_counters(
            core_ref_, state.device_counters(), state_size_, state.stream_id());
    }

    size_type state_size_{8};
    Collection<CoreStateCounters, Ownership::value, MemSpace::host> core_;
    CoreStateCounterRef<MemSpace::host> core_ref_;
};

TEST_F(GeneratorCounterStateTest, buffer_bound)
{
    GeneratorCounterState<MemSpace::host> state{StreamId{0}};
    EXPECT_EQ(0, state.buffer_bound());

    CountersT inserted;
    inserted.buffer_size = 5;
    state.assign(inserted);
    EXPECT_EQ(5, state.counters.buffer_size);
    EXPECT_EQ(5, state.buffer_bound());

    // Up to three items may be added in a step
    state.grow_bound(3);
    EXPECT_EQ(8, state.buffer_bound());

    // A completed copy tightens the bound to the actual size
    state.copy_async();
    EXPECT_EQ(5, state.buffer_bound());

    // Growth after the copy is kept in the bound
    state.copy_async();
    state.grow_bound(2);
    EXPECT_EQ(7, state.buffer_bound());
}

TEST_F(GeneratorCounterStateTest, generate)
{
    GeneratorCounterState<MemSpace::host> state{StreamId{0}};

    // Insert two distributions and count them when generation starts
    CountersT inserted;
    inserted.buffer_size = 2;
    state.assign(inserted);
    detail::accumulate_buffer(state.device_counters(), state.stream_id());
    size_type const total = 10;
    detail::update_pending(
        state.device_counters(), HostPtr{&total}, state.stream_id());

    // Generate into the four vacant slots
    this->set_core(4, 10);
    this->update(state);
    EXPECT_EQ(0, this->core().num_vacancies);
    EXPECT_EQ(6, this->core().num_pending);
    EXPECT_EQ(4, this->core().num_generated);
    EXPECT_EQ(state_size_, this->core().num_active);

    // The host copy is only updated by syncing
    EXPECT_EQ(0, state.counters.num_pending);
    state.sync_counters();
    EXPECT_EQ(2, state.counters.buffer_size);
    EXPECT_EQ(6, state.counters.num_pending);
    EXPECT_EQ(4, state.counters.num_generated);
    EXPECT_EQ(2, state.accum.buffer_size);
    EXPECT_EQ(10, state.accum.num_pending);
    EXPECT_EQ(4, state.accum.num_generated);

    // Generate the remaining photons, clearing the buffer but keeping the
    // number generated in this loop
    this->set_core(8, 6);
    this->update(state);
    EXPECT_EQ(2, this->core().num_vacancies);
    EXPECT_EQ(0, this->core().num_pending);
    EXPECT_EQ(6, this->core().num_active);

    state.sync_counters();
    EXPECT_TRUE(state.counters.empty());
    EXPECT_EQ(10, state.counters.num_generated);
    EXPECT_EQ(2, state.accum.buffer_size);
    EXPECT_EQ(10, state.accum.num_pending);
    EXPECT_EQ(10, state.accum.num_generated);

    // Increments are only accumulated once
    state.sync_counters();
    EXPECT_EQ(2, state.accum.buffer_size);
    EXPECT_EQ(10, state.accum.num_pending);
    EXPECT_EQ(10, state.accum.num_generated);

    // The generated count is cleared at the start of the next loop
    state.clear_generated();
    EXPECT_EQ(0, state.counters.num_generated);
    state.sync_counters();
    EXPECT_EQ(0, state.counters.num_generated);
    EXPECT_EQ(10, state.accum.num_generated);
}

TEST_F(GeneratorCounterStateTest, compact)
{
    GeneratorCounterState<MemSpace::host> state{StreamId{0}};
    auto set_size = [&state](size_type num_selected) {
        detail::set_buffer_size(state.device_counters(),
                                HostPtr{&num_selected},
                                state.stream_id());
    };
    auto set_pending = [&state](size_type total) {
        detail::update_pending(
            state.device_counters(), HostPtr{&total}, state.stream_id());
    };

    // Three distributions with 12 photons are appended to an empty buffer
    set_size(3);
    set_pending(12);

    // Removing distributions after generating doesn't change the statistics
    set_size(1);
    set_pending(12);

    // Two more distributions with 5 photons are appended
    set_size(3);
    set_pending(17);

    state.sync_counters();
    EXPECT_EQ(3, state.counters.buffer_size);
    EXPECT_EQ(17, state.counters.num_pending);
    EXPECT_EQ(5, state.accum.buffer_size);
    EXPECT_EQ(17, state.accum.num_pending);
    EXPECT_EQ(0, state.accum.num_generated);
}

TEST_F(GeneratorCounterStateTest, reset)
{
    GeneratorCounterState<MemSpace::host> state{StreamId{0}};

    CountersT inserted;
    inserted.buffer_size = 3;
    state.assign(inserted);
    detail::accumulate_buffer(state.device_counters(), state.stream_id());
    state.reset_counters();
    EXPECT_TRUE(state.counters.empty());
    EXPECT_EQ(0, state.buffer_bound());

    // The state copy was also cleared, but the queued data is still counted
    state.sync_counters();
    EXPECT_TRUE(state.counters.empty());
    EXPECT_EQ(3, state.accum.buffer_size);
    EXPECT_EQ(0, state.accum.num_generated);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
