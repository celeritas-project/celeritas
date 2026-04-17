//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/DetectorSteps.hh"

#include "corecel/data/ParamsDataStore.hh"
#include "corecel/data/Ref.hh"
#include "celeritas/user/StepData.hh"
#include "celeritas/user/detail/StepScratchCopyExecutor.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
template<class C>
std::vector<int> extract_ids(C const& ids)
{
    std::vector<int> result(ids.size());
    std::transform(ids.begin(),
                   ids.end(),
                   result.begin(),
                   [](typename C::value_type const& v) {
                       return v ? v.unchecked_get() : -1;
                   });
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

class DetectorStepsTest : public ::celeritas::test::Test
{
  protected:
    using HostStates = StepStateData<Ownership::value, MemSpace::host>;
    using DeviceStates = StepStateData<Ownership::value, MemSpace::device>;
    using HostParamsRef = HostCRef<StepParamsData>;

  protected:
    void SetUp() override
    {
        // Construct params
        celeritas::HostVal<StepParamsData> host_data;

        // Four volumes, three detectors
        std::vector<DetectorId> detectors
            = {DetectorId{}, DetectorId{2}, DetectorId{1}, DetectorId{0}};
        make_builder(&host_data.detector)
            .insert_back(detectors.begin(), detectors.end());

        host_data.selection = this->selection();
        host_data.num_volume_levels = 4;

        params_ = ParamsDataStore<StepParamsData>(std::move(host_data));

        if (auto& d = celeritas::device())
        {
            // Copies happen asynchronously
            d.create_streams(1);
        }
    }

    // Select all attributes by default
    virtual StepSelection selection() const
    {
        StepSelection result;
        for (auto& sp : result.points)
        {
            sp.time = true;
            sp.pos = true;
            sp.dir = true;
            sp.energy = true;
            sp.volume_id = true;
            sp.volume_instance_ids = true;
        }
        result.event_id = true;
        result.track_step_count = true;
        result.action_id = true;
        result.step_length = true;
        result.weight = true;
        result.particle = true;
        result.energy_deposition = true;
        return result;
    }

    HostParamsRef params() const { return params_.host_ref(); }

    HostStates build_states(size_type count)
    {
        CELER_EXPECT(count > 0);
        HostStates result;
        resize(&result, this->params(), StreamId{0}, count);
        auto& step = result.data;
        result.num_volume_levels = this->params().num_volume_levels;

        // Fill with bogus data
        int i = 0;
        for (auto tid : range(TrackSlotId{result.size()}))
        {
            for (auto sp : range(StepPoint::size_))
            {
                auto& state_point = step.points[sp];
                if (!state_point.time.empty())
                    state_point.time[tid] = i++;
                if (!state_point.pos.empty())
                    state_point.pos[tid] = Real3{real_type(i++), 1, 2};
                if (!state_point.dir.empty())
                    state_point.dir[tid] = Real3{real_type(i++), 10, 20};
                if (!state_point.volume_id.empty())
                    state_point.volume_id[tid] = ImplVolumeId(i++ % 4);
                if (!state_point.energy.empty())
                    state_point.energy[tid] = units::MevEnergy(i++);

                if (!state_point.volume_instance_ids.empty())
                {
                    using ViId = ItemId<VolumeInstanceId>;
                    auto depth = tid.unchecked_get() % 4;
                    for (auto j : range(result.num_volume_levels))
                    {
                        VolumeInstanceId val;
                        if (j <= depth)
                        {
                            val = id_cast<VolumeInstanceId>((j + i) % 8);
                        }
                        ViId vi_id{
                            result.num_volume_levels * tid.unchecked_get() + j};
                        state_point.volume_instance_ids[vi_id] = val;
                    }
                }
            }
            // Leave occasional gaps in the track IDs
            step.track_id[tid] = tid.get() % 5 == 0 ? TrackId{} : TrackId(i++);

            // Cycle through detector ids
            DetectorId det{tid.get() % 4};
            if (!step.track_id[tid] || det == DetectorId{3})
                det = {};
            step.detector[tid] = det;

            if (!step.event_id.empty())
                step.event_id[tid] = EventId(i++);
            if (!step.track_step_count.empty())
                step.track_step_count[tid] = i++;
            if (!step.action_id.empty())
                step.action_id[tid] = ActionId(i++);
            if (!step.step_length.empty())
                step.step_length[tid] = i++;
            if (!step.weight.empty())
                step.weight[tid] = 0.9;
            if (!step.particle.empty())
                step.particle[tid] = ParticleId(i++);
            if (!step.energy_deposition.empty())
                step.energy_deposition[tid] = units::MevEnergy(i++);
        }

        return result;
    }

  private:
    ParamsDataStore<StepParamsData> params_;
};

class SmallDetectorStepsTest : public DetectorStepsTest
{
  public:
    StepSelection selection() const override
    {
        StepSelection result;
        result.points[StepPoint::pre].pos = true;
        result.points[StepPoint::post].pos = true;
        result.energy_deposition = true;
        return result;
    }
};

//---------------------------------------------------------------------------//

TEST_F(DetectorStepsTest, host)
{
    auto states = this->build_states(32);

    // Create output placeholder and copy data over
    DetectorStepOutput output;
    copy_steps(&output, make_ref(states));

    static int const expected_detector[]
        = {1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1};
    EXPECT_VEC_EQ(expected_detector, extract_ids(output.detector));

    std::size_t num_tracks = 18;
    EXPECT_EQ(num_tracks, output.track_id.size());
    EXPECT_EQ(num_tracks, output.event_id.size());
    EXPECT_EQ(num_tracks, output.track_step_count.size());
    EXPECT_EQ(num_tracks, output.step_length.size());
    EXPECT_EQ(num_tracks, output.weight.size());
    EXPECT_EQ(num_tracks, output.particle.size());
    EXPECT_EQ(num_tracks, output.energy_deposition.size());

    auto const& pre = output.points[StepPoint::pre];
    EXPECT_EQ(num_tracks, pre.time.size());
    EXPECT_EQ(num_tracks, pre.pos.size());
    EXPECT_EQ(num_tracks, pre.dir.size());
    EXPECT_EQ(num_tracks, pre.energy.size());

    static int const expected_pre_volume_instance_ids[] = {
        5,  6,  -1, -1, 6,  7,  0,  -1, 0,  -1, -1, -1, 1,  2,  3,  -1, 3,  -1,
        -1, -1, 4,  5,  -1, -1, 6,  -1, -1, -1, 7,  0,  -1, -1, 0,  1,  2,  -1,
        1,  -1, -1, -1, 2,  3,  -1, -1, 3,  4,  5,  -1, 5,  6,  -1, -1, 6,  7,
        0,  -1, 0,  -1, -1, -1, 1,  2,  3,  -1, 3,  -1, -1, -1, 4,  5,  -1, -1,
    };
    EXPECT_VEC_EQ(expected_pre_volume_instance_ids,
                  extract_ids(pre.volume_instance_ids));

    auto const& post = output.points[StepPoint::post];
    EXPECT_EQ(num_tracks, post.time.size());
    EXPECT_EQ(num_tracks, post.pos.size());
    EXPECT_EQ(num_tracks, post.dir.size());
    EXPECT_EQ(num_tracks, post.energy.size());

    static int const expected_post_volume_instance_ids[] = {
        2,  3,  -1, -1, 3,  4,  5,  -1, 5,  -1, -1, -1, 6,  7,  0,  -1, 0,  -1,
        -1, -1, 1,  2,  -1, -1, 3,  -1, -1, -1, 4,  5,  -1, -1, 5,  6,  7,  -1,
        6,  -1, -1, -1, 7,  0,  -1, -1, 0,  1,  2,  -1, 2,  3,  -1, -1, 3,  4,
        5,  -1, 5,  -1, -1, -1, 6,  7,  0,  -1, 0,  -1, -1, -1, 1,  2,  -1, -1,
    };
    EXPECT_VEC_EQ(expected_post_volume_instance_ids,
                  extract_ids(post.volume_instance_ids));
}

TEST_F(DetectorStepsTest, TEST_IF_CELER_DEVICE(device))
{
    size_type constexpr num_tracks = 300;

    DeviceStates device_states;
    resize(&device_states, this->params(), StreamId{0}, num_tracks);
    auto host_states = this->build_states(num_tracks);
    device_states.data = host_states.data;
    ASSERT_EQ(num_tracks, device_states.size());
    ASSERT_TRUE(static_cast<bool>(device_states));

    // Construct reference values
    DetectorStepOutput host_output;
    copy_steps(&host_output, make_ref(host_states));

    // Perform reduction on device and copy back to host
    DetectorStepOutput output;
    copy_steps(&output, make_ref(device_states));

    EXPECT_VEC_EQ(host_output.track_id, output.track_id);
    EXPECT_VEC_EQ(host_output.event_id, output.event_id);
    EXPECT_VEC_EQ(host_output.track_step_count, output.track_step_count);
    EXPECT_VEC_EQ(host_output.step_length, output.step_length);
    EXPECT_VEC_EQ(host_output.weight, output.weight);
    EXPECT_VEC_EQ(host_output.particle, output.particle);
    EXPECT_VEC_EQ(host_output.energy_deposition, output.energy_deposition);

    auto const& host_pre = host_output.points[StepPoint::pre];
    auto const& pre = output.points[StepPoint::pre];
    EXPECT_VEC_EQ(host_pre.time, pre.time);
    EXPECT_VEC_EQ(host_pre.pos, pre.pos);
    EXPECT_VEC_EQ(host_pre.dir, pre.dir);
    EXPECT_VEC_EQ(host_pre.energy, pre.energy);
    EXPECT_VEC_EQ(host_pre.volume_instance_ids, pre.volume_instance_ids);

    auto const& host_post = host_output.points[StepPoint::post];
    auto const& post = output.points[StepPoint::post];
    EXPECT_VEC_EQ(host_post.time, post.time);
    EXPECT_VEC_EQ(host_post.pos, post.pos);
    EXPECT_VEC_EQ(host_post.dir, post.dir);
    EXPECT_VEC_EQ(host_post.energy, post.energy);
    EXPECT_VEC_EQ(host_post.volume_instance_ids, post.volume_instance_ids);
}

TEST_F(SmallDetectorStepsTest, host)
{
    auto states = this->build_states(32);

    // Create output placeholder and copy data over
    DetectorStepOutput output;
    copy_steps(&output, make_ref(states));

    static int const expected_detector[]
        = {1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1};
    EXPECT_VEC_EQ(expected_detector, extract_ids(output.detector));

    std::size_t num_tracks = 18;
    EXPECT_EQ(num_tracks, output.track_id.size());
    EXPECT_EQ(0, output.event_id.size());
    EXPECT_EQ(0, output.track_step_count.size());
    EXPECT_EQ(0, output.step_length.size());
    EXPECT_EQ(0, output.weight.size());
    EXPECT_EQ(0, output.particle.size());
    EXPECT_EQ(num_tracks, output.energy_deposition.size());

    auto const& pre = output.points[StepPoint::pre];
    EXPECT_EQ(0, pre.time.size());
    EXPECT_EQ(num_tracks, pre.pos.size());
    EXPECT_EQ(0, pre.dir.size());
    EXPECT_EQ(0, pre.energy.size());
    EXPECT_EQ(0, pre.volume_instance_ids.size());

    auto const& post = output.points[StepPoint::post];
    EXPECT_EQ(0, post.time.size());
    EXPECT_EQ(num_tracks, post.pos.size());
    EXPECT_EQ(0, post.dir.size());
    EXPECT_EQ(0, post.energy.size());
    EXPECT_EQ(0, post.volume_instance_ids.size());
}

TEST_F(SmallDetectorStepsTest, TEST_IF_CELER_DEVICE(device))
{
    DeviceStates device_states;
    {
        size_type constexpr num_tracks = 1024;

        // Create states on host and copy to device
        resize(&device_states, this->params(), StreamId{0}, num_tracks);
        auto host_states = this->build_states(num_tracks);
        device_states.data = host_states.data;
        ASSERT_EQ(num_tracks, device_states.size());
        ASSERT_TRUE(static_cast<bool>(device_states));
    }

    // Perform reduction on device and copy back to host
    DetectorStepOutput output;
    copy_steps(&output, make_ref(device_states));

    std::size_t num_tracks = 614;
    EXPECT_EQ(num_tracks, output.track_id.size());
    EXPECT_EQ(0, output.event_id.size());
    EXPECT_EQ(0, output.track_step_count.size());
    EXPECT_EQ(0, output.step_length.size());
    EXPECT_EQ(0, output.weight.size());
    EXPECT_EQ(0, output.particle.size());
    EXPECT_EQ(num_tracks, output.energy_deposition.size());

    auto const& pre = output.points[StepPoint::pre];
    EXPECT_EQ(0, pre.time.size());
    EXPECT_EQ(num_tracks, pre.pos.size());
    EXPECT_EQ(0, pre.dir.size());
    EXPECT_EQ(0, pre.energy.size());
    EXPECT_EQ(0, pre.volume_instance_ids.size());

    auto const& post = output.points[StepPoint::post];
    EXPECT_EQ(0, post.time.size());
    EXPECT_EQ(num_tracks, post.pos.size());
    EXPECT_EQ(0, post.dir.size());
    EXPECT_EQ(0, post.energy.size());
    EXPECT_EQ(0, post.volume_instance_ids.size());
}

TEST_F(DetectorStepsTest, death_fields_allocated)
{
    // Build params with track_death enabled (no volume instances needed)
    celeritas::HostVal<StepParamsData> host_data;
    std::vector<DetectorId> detectors
        = {DetectorId{}, DetectorId{1}, DetectorId{0}};
    make_builder(&host_data.detector)
        .insert_back(detectors.begin(), detectors.end());
    host_data.selection = this->selection();
    for (auto& sp : host_data.selection.points)
    {
        sp.volume_instance_ids = false;
    }
    host_data.track_death = true;
    host_data.num_volume_levels = 0;

    auto params = ParamsDataStore<StepParamsData>(std::move(host_data));

    HostStates states;
    resize(&states, params.host_ref(), StreamId{0}, 16);

    // Death fields must be allocated
    EXPECT_EQ(16, states.data.death_track_id.size());
    EXPECT_EQ(16, states.data.death_primary_id.size());
    EXPECT_EQ(16, states.data.death_particle.size());
    EXPECT_EQ(16, states.data.death_pos.size());
    EXPECT_EQ(16, states.data.death_dir.size());
    EXPECT_EQ(16, states.data.death_energy.size());
    EXPECT_EQ(16, states.data.death_time.size());
}

TEST_F(DetectorStepsTest, death_fields_not_allocated)
{
    // Default params: track_death is false
    auto states = this->build_states(16);

    // Death fields must NOT be allocated
    EXPECT_EQ(0, states.data.death_track_id.size());
    EXPECT_EQ(0, states.data.death_primary_id.size());
    EXPECT_EQ(0, states.data.death_particle.size());
    EXPECT_EQ(0, states.data.death_pos.size());
    EXPECT_EQ(0, states.data.death_dir.size());
    EXPECT_EQ(0, states.data.death_energy.size());
    EXPECT_EQ(0, states.data.death_time.size());
}

TEST_F(DetectorStepsTest, copy_deaths_host)
{
    // Build params with track_death enabled (no volume instances needed)
    celeritas::HostVal<StepParamsData> host_data;
    std::vector<DetectorId> detectors
        = {DetectorId{}, DetectorId{1}, DetectorId{0}};
    make_builder(&host_data.detector)
        .insert_back(detectors.begin(), detectors.end());
    host_data.selection = this->selection();
    for (auto& sp : host_data.selection.points)
    {
        sp.volume_instance_ids = false;
    }
    host_data.track_death = true;
    host_data.num_volume_levels = 0;

    auto params = ParamsDataStore<StepParamsData>(std::move(host_data));

    HostStates states;
    resize(&states, params.host_ref(), StreamId{0}, 8);

    // Populate death fields: slots 1, 3, 5 have valid deaths
    auto& d = states.data;
    for (auto tid : range(TrackSlotId{8}))
    {
        d.death_track_id[tid] = TrackId{};
        d.death_primary_id[tid] = PrimaryId{};
        d.death_particle[tid] = ParticleId{};
        d.death_energy[tid] = units::MevEnergy{0};
        d.death_time[tid] = 0;
        d.death_pos[tid] = {0, 0, 0};
        d.death_dir[tid] = {0, 0, 1};
    }

    d.death_track_id[TrackSlotId{1}] = TrackId{10};
    d.death_primary_id[TrackSlotId{1}] = PrimaryId{0};
    d.death_particle[TrackSlotId{1}] = ParticleId{1};
    d.death_energy[TrackSlotId{1}] = units::MevEnergy{100};
    d.death_time[TrackSlotId{1}] = 1.5;
    d.death_pos[TrackSlotId{1}] = {1, 2, 3};
    d.death_dir[TrackSlotId{1}] = {0, 0, 1};

    d.death_track_id[TrackSlotId{3}] = TrackId{30};
    d.death_primary_id[TrackSlotId{3}] = PrimaryId{1};
    d.death_particle[TrackSlotId{3}] = ParticleId{0};
    d.death_energy[TrackSlotId{3}] = units::MevEnergy{200};
    d.death_time[TrackSlotId{3}] = 2.5;
    d.death_pos[TrackSlotId{3}] = {4, 5, 6};
    d.death_dir[TrackSlotId{3}] = {1, 0, 0};

    d.death_track_id[TrackSlotId{5}] = TrackId{50};
    d.death_primary_id[TrackSlotId{5}] = PrimaryId{2};
    d.death_particle[TrackSlotId{5}] = ParticleId{2};
    d.death_energy[TrackSlotId{5}] = units::MevEnergy{300};
    d.death_time[TrackSlotId{5}] = 3.5;
    d.death_pos[TrackSlotId{5}] = {7, 8, 9};
    d.death_dir[TrackSlotId{5}] = {0, 1, 0};

    // Compact deaths
    DetectorStepOutput output;
    copy_deaths(&output, make_ref(states));

    // Verify 3 deaths compacted
    ASSERT_EQ(3, output.deaths.size());

    EXPECT_EQ(TrackId{10}, output.deaths[0].track_id);
    EXPECT_EQ(PrimaryId{0}, output.deaths[0].primary_id);
    EXPECT_EQ(ParticleId{1}, output.deaths[0].particle);
    EXPECT_DOUBLE_EQ(100, output.deaths[0].final_energy.value());
    EXPECT_DOUBLE_EQ(1.5, output.deaths[0].final_time);
    EXPECT_EQ((Real3{1, 2, 3}), output.deaths[0].final_pos);

    EXPECT_EQ(TrackId{30}, output.deaths[1].track_id);
    EXPECT_EQ(PrimaryId{1}, output.deaths[1].primary_id);
    EXPECT_EQ(ParticleId{0}, output.deaths[1].particle);
    EXPECT_DOUBLE_EQ(200, output.deaths[1].final_energy.value());

    EXPECT_EQ(TrackId{50}, output.deaths[2].track_id);
    EXPECT_EQ(PrimaryId{2}, output.deaths[2].primary_id);
    EXPECT_EQ(ParticleId{2}, output.deaths[2].particle);
    EXPECT_DOUBLE_EQ(300, output.deaths[2].final_energy.value());
}

TEST_F(DetectorStepsTest, copy_deaths_empty)
{
    // Without track_death, copy_deaths should clear output
    auto states = this->build_states(8);

    DetectorStepOutput output;
    output.deaths.resize(5);  // Pre-populate to verify it gets cleared
    copy_deaths(&output, make_ref(states));

    EXPECT_TRUE(output.deaths.empty());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
