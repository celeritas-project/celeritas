//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/user/DetectorSteps.hh"

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/Ref.hh"
#include "celeritas/user/StepData.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
template<class V, class S>
std::vector<int> extract_ids(const std::vector<OpaqueId<V, S>>& ids)
{
    std::vector<int> result(ids.size());
    std::transform(
        ids.begin(), ids.end(), result.begin(), [](const OpaqueId<V, S>& v) {
            return v ? v.unchecked_get() : -1;
        });
    return result;
}

//---------------------------------------------------------------------------//
} // namespace

class DetectorStepsTest : public ::celeritas::test::Test
{
  protected:
    using HostStates = StepStateData<Ownership::value, MemSpace::host>;

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

        params_ = CollectionMirror<StepParamsData>(std::move(host_data));
    }

    // Select all attributes by default
    virtual StepSelection selection() const
    {
        StepSelection result;
        for (auto& sp : result.points)
        {
            sp.time   = true;
            sp.pos    = true;
            sp.dir    = true;
            sp.volume = true;
            sp.energy = true;
        }
        result.event             = true;
        result.track_step_count  = true;
        result.action            = true;
        result.step_length       = true;
        result.particle          = true;
        result.energy_deposition = true;
        return result;
    }

    HostStates build_states(size_type count)
    {
        CELER_EXPECT(count > 0);
        HostStates result;
        resize(&result, params_.host_ref(), count);

        // Fill with bogus data
        int i = 0;
        for (auto tid : range(ThreadId{result.size()}))
        {
            for (auto sp : range(StepPoint::size_))
            {
                auto& state_point = result.points[sp];
                if (!state_point.time.empty())
                    state_point.time[tid] = i++;
                if (!state_point.pos.empty())
                    state_point.pos[tid] = Real3{real_type(i++), 1, 2};
                if (!state_point.dir.empty())
                    state_point.dir[tid] = Real3{real_type(i++), 10, 20};
                if (!state_point.dir.empty())
                    state_point.volume[tid] = VolumeId(i++ % 4);
                if (!state_point.energy.empty())
                    state_point.energy[tid] = units::MevEnergy(i++);
            }
            // Leave occasional gaps in the track IDs
            result.track[tid] = tid.get() % 5 == 0 ? TrackId{} : TrackId(i++);

            // Cycle through detector ids
            DetectorId det{tid.get() % 4};
            if (!result.track[tid] || det == DetectorId{3})
                det = {};
            result.detector[tid] = det;

            if (!result.event.empty())
                result.event[tid] = EventId(i++);
            if (!result.track_step_count.empty())
                result.track_step_count[tid] = i++;
            if (!result.action.empty())
                result.action[tid] = ActionId(i++);
            if (!result.step_length.empty())
                result.step_length[tid] = i++;

            if (!result.particle.empty())
                result.particle[tid] = ParticleId(i++);
            if (!result.energy_deposition.empty())
                result.energy_deposition[tid] = units::MevEnergy(i++);
        }

        return result;
    }

  private:
    CollectionMirror<StepParamsData> params_;
};

class SmallDetectorStepsTest : public DetectorStepsTest
{
  public:
    StepSelection selection() const override
    {
        StepSelection result;
        result.points[StepPoint::pre].pos  = true;
        result.points[StepPoint::post].pos = true;
        result.energy_deposition           = true;
        return result;
    }
};

//---------------------------------------------------------------------------//

TEST_F(DetectorStepsTest, host)
{
    auto states = this->build_states(32);

    // Create output placeholder and copy data over
    DetectorStepOutput output;
    copy(&output, make_ref(states));

    static const int expected_detector[]
        = {1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1};
    EXPECT_VEC_EQ(expected_detector, extract_ids(output.detector));

    std::size_t num_tracks = 18;
    EXPECT_EQ(num_tracks, output.track.size());
    EXPECT_EQ(num_tracks, output.event.size());
    EXPECT_EQ(num_tracks, output.track_step_count.size());
    EXPECT_EQ(num_tracks, output.step_length.size());
    EXPECT_EQ(num_tracks, output.particle.size());
    EXPECT_EQ(num_tracks, output.energy_deposition.size());

    const auto& pre = output.points[StepPoint::pre];
    EXPECT_EQ(num_tracks, pre.time.size());
    EXPECT_EQ(num_tracks, pre.pos.size());
    EXPECT_EQ(num_tracks, pre.dir.size());
    EXPECT_EQ(num_tracks, pre.energy.size());

    const auto& post = output.points[StepPoint::post];
    EXPECT_EQ(num_tracks, post.time.size());
    EXPECT_EQ(num_tracks, post.pos.size());
    EXPECT_EQ(num_tracks, post.dir.size());
    EXPECT_EQ(num_tracks, post.energy.size());
}

TEST_F(DetectorStepsTest, TEST_IF_CELER_DEVICE(device))
{
    CollectionStateStore<StepStateData, MemSpace::device> device_states;
    {
        // Create states on host and copy to device
        auto host_states = this->build_states(300);
        device_states    = host_states;
    }
    ASSERT_EQ(300, device_states.size());

    // Perform reduction on device and copy back to host
    DetectorStepOutput output;
    copy(&output, device_states.ref());

    // Check a subset of the detector IDs
    auto det_ids = extract_ids(output.detector);
    det_ids.erase(det_ids.begin() + std::min<std::size_t>(det_ids.size(), 18),
                  det_ids.end());
    static const int expected_detector[]
        = {1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1};
    EXPECT_VEC_EQ(expected_detector, det_ids);

    std::size_t num_tracks = 180;
    EXPECT_EQ(num_tracks, output.track.size());
    EXPECT_EQ(num_tracks, output.event.size());
    EXPECT_EQ(num_tracks, output.track_step_count.size());
    EXPECT_EQ(num_tracks, output.step_length.size());
    EXPECT_EQ(num_tracks, output.particle.size());
    EXPECT_EQ(num_tracks, output.energy_deposition.size());

    const auto& pre = output.points[StepPoint::pre];
    EXPECT_EQ(num_tracks, pre.time.size());
    EXPECT_EQ(num_tracks, pre.pos.size());
    EXPECT_EQ(num_tracks, pre.dir.size());
    EXPECT_EQ(num_tracks, pre.energy.size());

    const auto& post = output.points[StepPoint::post];
    EXPECT_EQ(num_tracks, post.time.size());
    EXPECT_EQ(num_tracks, post.pos.size());
    EXPECT_EQ(num_tracks, post.dir.size());
    EXPECT_EQ(num_tracks, post.energy.size());
}

TEST_F(SmallDetectorStepsTest, host)
{
    auto states = this->build_states(32);

    // Create output placeholder and copy data over
    DetectorStepOutput output;
    copy(&output, make_ref(states));

    static const int expected_detector[]
        = {1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1};
    EXPECT_VEC_EQ(expected_detector, extract_ids(output.detector));

    std::size_t num_tracks = 18;
    EXPECT_EQ(num_tracks, output.track.size());
    EXPECT_EQ(0, output.event.size());
    EXPECT_EQ(0, output.track_step_count.size());
    EXPECT_EQ(0, output.step_length.size());
    EXPECT_EQ(0, output.particle.size());
    EXPECT_EQ(num_tracks, output.energy_deposition.size());

    const auto& pre = output.points[StepPoint::pre];
    EXPECT_EQ(0, pre.time.size());
    EXPECT_EQ(num_tracks, pre.pos.size());
    EXPECT_EQ(0, pre.dir.size());
    EXPECT_EQ(0, pre.energy.size());

    const auto& post = output.points[StepPoint::post];
    EXPECT_EQ(0, post.time.size());
    EXPECT_EQ(num_tracks, post.pos.size());
    EXPECT_EQ(0, post.dir.size());
    EXPECT_EQ(0, post.energy.size());
}

TEST_F(SmallDetectorStepsTest, TEST_IF_CELER_DEVICE(device))
{
    CollectionStateStore<StepStateData, MemSpace::device> device_states;
    {
        // Create states on host and copy to device
        auto host_states = this->build_states(1024);
        device_states    = host_states;
    }

    // Perform reduction on device and copy back to host
    DetectorStepOutput output;
    copy(&output, device_states.ref());

    std::size_t num_tracks = 614;
    EXPECT_EQ(num_tracks, output.track.size());
    EXPECT_EQ(0, output.event.size());
    EXPECT_EQ(0, output.track_step_count.size());
    EXPECT_EQ(0, output.step_length.size());
    EXPECT_EQ(0, output.particle.size());
    EXPECT_EQ(num_tracks, output.energy_deposition.size());

    const auto& pre = output.points[StepPoint::pre];
    EXPECT_EQ(0, pre.time.size());
    EXPECT_EQ(num_tracks, pre.pos.size());
    EXPECT_EQ(0, pre.dir.size());
    EXPECT_EQ(0, pre.energy.size());

    const auto& post = output.points[StepPoint::post];
    EXPECT_EQ(0, post.time.size());
    EXPECT_EQ(num_tracks, post.pos.size());
    EXPECT_EQ(0, post.dir.size());
    EXPECT_EQ(0, post.energy.size());
}

//---------------------------------------------------------------------------//
} // namespace test
} // namespace celeritas
