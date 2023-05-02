//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/TrackInit.test.cc
//---------------------------------------------------------------------------//
#include "TrackInit.test.hh"

#include <algorithm>
#include <numeric>
#include <vector>

#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/io/LogContextException.hh"
#include "celeritas/SimpleTestBase.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/global/ActionRegistry.hh"
#include "celeritas/global/CoreParams.hh"
#include "celeritas/global/CoreTrackData.hh"
#include "celeritas/global/detail/ActionSequence.hh"
#include "celeritas/track/ExtendFromSecondariesAction.hh"
#include "celeritas/track/InitializeTracksAction.hh"
#include "celeritas/track/TrackInitParams.hh"
#include "celeritas/track/TrackInitUtils.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
using TrackInitDeviceValue
    = TrackInitStateData<Ownership::value, MemSpace::device>;

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//

ITTestInput::ITTestInput(std::vector<size_type>& host_alloc_size,
                         std::vector<char>& host_alive)
    : alloc_size(host_alloc_size.size()), alive(host_alive.size())
{
    CELER_EXPECT(host_alloc_size.size() == host_alive.size());
    alloc_size.copy_to_device(make_span(host_alloc_size));
    alive.copy_to_device(make_span(host_alive));
}

ITTestInputData ITTestInput::device_ref() const
{
    ITTestInputData result;
    result.alloc_size = alloc_size.device_ref();
    result.alive = alive.device_ref();
    return result;
}

//! Wrap interact kernel in an post step Action
class InteractAction final : public ExplicitActionInterface
{
  public:
    InteractAction(ActionId id,
                   std::vector<size_type> alloc,
                   std::vector<char> alive)
        : id_(id), input_(alloc, alive)
    {
    }

    // Launch kernel with host data, should never be called
    void execute(ParamsHostCRef const&, StateHostRef&) const final
    {
        CELER_NOT_IMPLEMENTED("InteractAction is device-only");
    };

    void
    execute(ParamsDeviceCRef const& params, StateDeviceRef& states) const final
    {
        interact(params, states, input_.device_ref());
    }

    ActionId action_id() const final { return id_; }
    std::string label() const final { return "interact"; }
    std::string description() const final { return "interact kernel"; }

    ActionOrder order() const final { return ActionOrder::along; }

  private:
    ActionId id_;
    ITTestInput input_;
};

//! Copy results to host
ITTestOutput get_result(DeviceRef<CoreStateData>& states)
{
    CELER_EXPECT(states);

    ITTestOutput result;

    // Copy track initializer data to host
    HostVal<TrackInitStateData> data;
    data = states.init;

    // Store the IDs of the vacant track slots
    for (TrackSlotId const& v : data.vacancies.data())
    {
        result.vacancies.push_back(v.unchecked_get());
    }

    // Store the track IDs of the initializers
    for (auto const& init : data.initializers.data())
    {
        result.init_ids.push_back(init.sim.track_id.get());
    }

    // Copy sim states to host
    SimStateData<Ownership::value, MemSpace::host> sim;
    sim = states.sim;

    // Store the track IDs and parent IDs
    for (auto tid : range(TrackSlotId{sim.size()}))
    {
        result.track_ids.push_back(sim.track_ids[tid].unchecked_get());
        result.parent_ids.push_back(sim.parent_ids[tid].unchecked_get());
    }

    return result;
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

#define TrackInitTest TEST_IF_CELER_DEVICE(TrackInitTest)

class TrackInitTest : public SimpleTestBase
{
  protected:
    void SetUp() override
    {
        core_params = this->core()->device_ref();
        CELER_ENSURE(core_params);
    }

    //! Create primary particles
    std::vector<Primary> generate_primaries(size_type num_primaries)
    {
        std::vector<Primary> result;
        for (unsigned int i = 0; i < num_primaries; ++i)
        {
            Primary p;
            p.particle_id = ParticleId{0};
            p.energy = units::MevEnergy{1. + i};
            p.position = {0, 0, 0};
            p.direction = {0, 0, 1};
            p.time = 0;
            p.event_id = EventId{0};
            p.track_id = TrackId{i};
            result.push_back(p);
        }
        return result;
    }

    //! Create mutable state data
    void build_states(size_type num_tracks)
    {
        CELER_EXPECT(core_params);

        // Allocate state data
        resize(
            &device_states, this->core()->host_ref(), StreamId{0}, num_tracks);
        core_state = device_states;

        CELER_ENSURE(core_state);
    }

    //! Copy results to host
    ITTestOutput get_result(DeviceRef<CoreStateData>& states)
    {
        CELER_EXPECT(states);

        ITTestOutput result;

        // Copy track initializer data to host
        HostVal<TrackInitStateData> data;
        data = states.init;

        // Store the IDs of the vacant track slots
        for (TrackSlotId const& v : data.vacancies.data())
        {
            result.vacancies.push_back(v.unchecked_get());
        }

        // Store the track IDs of the initializers
        for (auto const& init : data.initializers.data())
        {
            result.init_ids.push_back(init.sim.track_id.get());
        }

        // Copy sim states to host
        SimStateData<Ownership::value, MemSpace::host> sim;
        sim = states.sim;

        // Store the track IDs and parent IDs
        for (auto tid : range(TrackSlotId{sim.size()}))
        {
            result.track_ids.push_back(sim.track_ids[tid].unchecked_get());
            result.parent_ids.push_back(sim.parent_ids[tid].unchecked_get());
        }

        return result;
    }

    CoreStateData<Ownership::value, MemSpace::device> device_states;
    DeviceCRef<CoreParamsData> core_params;
    DeviceRef<CoreStateData> core_state;
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(TrackInitTest, run)
{
    const size_type num_primaries = 12;
    const size_type num_tracks = 10;

    build_states(num_tracks);

    // Check that all of the track slots were marked as empty
    {
        auto result = get_result(core_state);
        static unsigned int const expected_vacancies[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Create track initializers on device from primary particles
    auto primaries = generate_primaries(num_primaries);
    extend_from_primaries(core_params, core_state, make_span(primaries));

    // Check the track IDs of the track initializers created from primaries
    {
        auto result = get_result(core_state);
        static unsigned int const expected_track_ids[]
            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.init_ids);
    }

    // Initialize the primary tracks on device
    initialize_tracks(core_params, core_state);

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = get_result(core_state);
        static unsigned int const expected_track_ids[]
            = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        EXPECT_VEC_EQ(expected_track_ids, result.track_ids);

        // All primary particles, so no parent
        static int const expected_parent_ids[]
            = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    }

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 0, 0, 1, 1, 0, 0, 2, 1};
    std::vector<char> alive = {0, 1, 0, 1, 0, 1, 0, 1, 0, 0};
    ITTestInput input(alloc, alive);

    // Launch kernel to process interactions
    interact(core_params, core_state, input.device_ref());

    // Launch a kernel to create track initializers from secondaries
    extend_from_secondaries(core_params, core_state);

    // Check the vacancies
    {
        auto result = get_result(core_state);
        static unsigned int const expected_vacancies[] = {2, 6};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    }

    // Check the track IDs of the track initializers created from secondaries.
    // Because IDs are not calculated deterministically and we don't know which
    // IDs were used for the immediately-initialized secondaries and which were
    // used for the track initializers, just check that there is the correct
    // number and they are in the correct range.
    {
        auto result = get_result(core_state);
        EXPECT_TRUE(std::all_of(std::begin(result.init_ids),
                                std::end(result.init_ids),
                                [](unsigned int id) { return id <= 18; }));
        EXPECT_EQ(5, result.init_ids.size());

        // First two initializers are from primaries
        EXPECT_EQ(0, result.init_ids[0]);
        EXPECT_EQ(1, result.init_ids[1]);
    }

    // Initialize secondaries on device
    initialize_tracks(core_params, core_state);

    // Check the track IDs and parent IDs of the initialized tracks
    {
        auto result = get_result(core_state);
        EXPECT_TRUE(std::all_of(std::begin(result.track_ids),
                                std::end(result.track_ids),
                                [](unsigned int id) { return id <= 18; }));

        // Tracks that were not killed should have the same ID
        EXPECT_EQ(3, result.track_ids[1]);
        EXPECT_EQ(5, result.track_ids[3]);
        EXPECT_EQ(7, result.track_ids[5]);
        EXPECT_EQ(9, result.track_ids[7]);

        // Two tracks should have the same parent ID = 10
        int expected_parent_ids[] = {2, -1, 7, -1, 6, -1, 10, -1, 10, 11};
        std::sort(std::begin(result.parent_ids), std::end(result.parent_ids));
        std::sort(std::begin(expected_parent_ids),
                  std::end(expected_parent_ids));
        EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    }
}

TEST_F(TrackInitTest, primaries)
{
    const size_type num_sets = 4;
    const size_type num_primaries = 16;
    const size_type num_tracks = 16;

    build_states(num_tracks);

    // Kill half the tracks in each interaction and don't produce secondaries
    std::vector<size_type> alloc(num_tracks, 0);
    std::vector<char> alive(num_tracks);
    for (size_type i = 0; i < num_tracks; ++i)
    {
        alive[i] = i % 2;
    }
    ITTestInput input(alloc, alive);

    for (size_type i = 0; i < num_sets; ++i)
    {
        // Create track initializers on device from primary particles
        auto primaries = generate_primaries(num_primaries);
        extend_from_primaries(core_params, core_state, make_span(primaries));

        // Initialize tracks on device
        initialize_tracks(core_params, core_state);

        // Launch kernel that will kill half the tracks
        interact(core_params, core_state, input.device_ref());

        // Find vacancies and create track initializers from secondaries
        extend_from_secondaries(core_params, core_state);
        EXPECT_EQ(i * num_tracks / 2, core_state.init.initializers.size());
        EXPECT_EQ(num_tracks / 2, core_state.init.vacancies.size());
    }

    // Check the results
    static unsigned int const expected_track_ids[] = {
        8u, 1u, 9u, 3u, 10u, 5u, 11u, 7u, 12u, 9u, 13u, 11u, 14u, 13u, 15u, 15u};
    static int const expected_parent_ids[]
        = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static unsigned int const expected_vacancies[]
        = {0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u};
    static unsigned int const expected_init_ids[]
        = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u,
           4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    auto result = get_result(core_state);
    EXPECT_VEC_EQ(expected_track_ids, result.track_ids);
    EXPECT_VEC_EQ(expected_parent_ids, result.parent_ids);
    EXPECT_VEC_EQ(expected_vacancies, result.vacancies);
    EXPECT_VEC_EQ(expected_init_ids, result.init_ids);
}

#define TrackInitSecondaryTest TEST_IF_CELER_DEVICE(TrackInitSecondaryTest)

class TrackInitSecondaryTest : public TrackInitTest
{
  protected:
    real_type secondary_stack_factor() const final { return 8; }
};

TEST_F(TrackInitSecondaryTest, secondaries)
{
    const size_type num_primaries = 512;
    const size_type num_tracks = 512;

    build_states(num_tracks);

    // Allocate input device data (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 2, 0, 0, 0, 0, 0};
    std::vector<char> alive = {1, 0, 0, 1, 1, 0, 0, 1};
    size_type base_size = alive.size();
    for (size_type i = 0; i < num_tracks / base_size - 1; ++i)
    {
        alloc.insert(alloc.end(), alloc.begin(), alloc.begin() + base_size);
        alive.insert(alive.end(), alive.begin(), alive.begin() + base_size);
    }
    ITTestInput input(alloc, alive);

    // Create track initializers on device from primary particles
    auto primaries = generate_primaries(num_primaries);
    extend_from_primaries(core_params, core_state, make_span(primaries));
    EXPECT_EQ(num_primaries, core_state.init.initializers.size());

    const size_type num_iter = 16;
    for ([[maybe_unused]] size_type i : range(num_iter))
    {
        // All queued initializers are converted to tracks
        initialize_tracks(core_params, core_state);
        ASSERT_EQ(0, core_state.init.initializers.size());

        // Launch kernel to process interactions
        interact(core_params, core_state, input.device_ref());

        // Launch a kernel to create track initializers from secondaries
        extend_from_secondaries(core_params, core_state);
        ASSERT_EQ(128, core_state.init.initializers.size())
            << "iteration " << i;
        ASSERT_EQ(128, core_state.init.vacancies.size());
    }
}

TEST_F(TrackInitSecondaryTest, secondaries_action)
{
    // Basic setup
    const size_type num_primaries = 8;
    const size_type num_tracks = 8;

    build_states(num_tracks);

    // Create Actions
    ActionRegistry action_reg;
    action_reg.insert(
        std::make_shared<InitializeTracksAction>(action_reg.next_id()));
    action_reg.insert(
        std::make_shared<ExtendFromSecondariesAction>(action_reg.next_id()));

    // Create Interaction Action (number of secondaries to produce for each
    // track and whether the track survives the interaction)
    std::vector<size_type> alloc = {1, 1, 2, 0, 0, 0, 0, 0};
    std::vector<char> alive = {1, 0, 0, 1, 1, 0, 0, 1};
    action_reg.insert(
        std::make_shared<InteractAction>(action_reg.next_id(), alloc, alive));

    detail::ActionSequence::Options opts;
    detail::ActionSequence actions_(action_reg, opts);

    // Create track initializers on device from primary particles
    // TODO: will eventually become an action.
    auto primaries = generate_primaries(num_primaries);
    extend_from_primaries(core_params, core_state, make_span(primaries));
    EXPECT_EQ(num_primaries, core_state.init.initializers.size());

    const size_type num_iter = 4;
    for ([[maybe_unused]] size_type i : range(num_iter))
    {
        CELER_TRY_HANDLE(actions_.execute(core_params, core_state),
                         log_context_exception);
        auto result = get_result(core_state);

        // Slots 5 and 6 are always vacant because these tracks are killed with
        // no secondaries
        static unsigned int const expected_vacancies[] = {5u, 6u};
        EXPECT_VEC_EQ(expected_vacancies, result.vacancies);

        // init ids may not be deterministic, but can guarantee they are in the
        // range 8<=x<=12 as we create 4 tracks per iteration, 2 in reused
        // slots from their parent, 2 as new inits
        EXPECT_EQ(2, result.init_ids.size());
        EXPECT_TRUE(std::all_of(std::begin(result.init_ids),
                                std::end(result.init_ids),
                                [i](unsigned int id) {
                                    return id >= 8 + i * 4 && id <= 11 + i * 4;
                                }));

        // Track ids may not be deterministic, so only validate size and
        // range. (Remember that we create 4 new tracks per iteration, with 2
        // slots reused
        EXPECT_EQ(num_tracks, result.track_ids.size());
        EXPECT_TRUE(std::all_of(
            std::begin(result.track_ids),
            std::end(result.track_ids),
            [i](unsigned int id) { return id < num_tracks + (i + 1) * 4; }));

        // Parent ids may not be deterministic, but all non-killed tracks are
        // guaranteed to be primaries at every iteration. At end of first
        // iteration, will still have some primary ids as these are not cleared
        // until the next iteration
        for (size_type pidx : range(num_tracks))
        {
            EXPECT_TRUE(((alive[pidx] == 1) ? result.parent_ids[pidx] == -1
                                            : result.parent_ids[pidx] >= -1))
                << "iteration " << i;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
