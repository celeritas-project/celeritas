//------------------------------ -*- cuda -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DetectorSteps.cu
//---------------------------------------------------------------------------//
#include "DetectorSteps.hh"

#include <vector>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "corecel/data/Collection.hh"
#include "corecel/data/Copier.hh"
#include "corecel/data/ObserverPtr.device.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/KernelLauncher.device.hh"
#include "corecel/sys/KernelParamCalculator.device.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/Stream.hh"
#include "corecel/sys/Thrust.device.hh"

#include "StepData.hh"

#include "detail/StepScratchCopyExecutor.hh"

using namespace celeritas::literals;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
template<class T>
using StateRef
    = celeritas::StateCollection<T, Ownership::reference, MemSpace::native>;

template<class T>
using ItemRef
    = celeritas::Collection<T, Ownership::reference, MemSpace::native>;

//---------------------------------------------------------------------------//
template<class T>
struct IsValid
{
    CELER_FORCEINLINE_FUNCTION bool operator()(T const& id)
    {
        return static_cast<bool>(id);
    }
};

//---------------------------------------------------------------------------//
size_type count_num_valid(
    StepStateData<Ownership::reference, MemSpace::device> const& state)
{
    // Store the thread IDs of active tracks that are in a detector
    auto start = device_pointer_cast(state.valid_id.data());
    auto end = thrust::copy_if(thrust_execute_on(state.stream_id),
                               thrust::make_counting_iterator(0_sz),
                               thrust::make_counting_iterator(state.size()),
                               device_pointer_cast(state.data.detector.data()),
                               start,
                               IsValid<DetectorId>{});
    return end - start;
}

//---------------------------------------------------------------------------//
template<class T>
void copy_field(DetectorStepOutput::PinnedVec<T>* dst,
                StateRef<T> const& src,
                size_type num_valid,
                StreamId stream)
{
    if (src.empty() || num_valid == 0)
    {
        // This field is not in use or had no hits
        dst->clear();
        return;
    }
    dst->resize(num_valid);
    // Copy all items from valid threads
    Copier<T, MemSpace::host> copy{{dst->data(), num_valid}, stream};
    copy(MemSpace::device, {src.data().get(), num_valid});
}

//---------------------------------------------------------------------------//
template<class T>
void copy_field(DetectorStepOutput::PinnedVec<T>* dst,
                ItemRef<T> const& src,
                size_type num_valid,
                size_type per_thread,
                StreamId stream)
{
    CELER_EXPECT(per_thread > 0 || src.empty());
    if (src.empty() || num_valid == 0)
    {
        // This attribute is not in use
        dst->clear();
        return;
    }
    dst->resize(num_valid * per_thread);
    // Copy all items from valid threads
    Copier<T, MemSpace::host> copy{{dst->data(), num_valid * per_thread},
                                   stream};
    copy(MemSpace::device, {src.data().get(), num_valid * per_thread});
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Copy to host results from tracks that interacted with a detector.
 */
template<>
void copy_steps<MemSpace::device>(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::device> const& state)
{
    CELER_EXPECT(output);

    ScopedProfiling profile_this{"copy-steps"};

    // Get the number of threads that are active and in a detector
    size_type const num_valid = count_num_valid(state);

    // Gather the step data on device
    {
        auto execute_thread = detail::StepScratchCopyExecutor{state, num_valid};
        static KernelLauncher<decltype(execute_thread)> const launch_kernel(
            "gather-step-scratch");
        launch_kernel(num_valid, state.stream_id, execute_thread);
    }

    // Resize and copy if the fields are present
#define DS_ASSIGN(FIELD) \
    copy_field(          \
        &(output->FIELD), state.scratch.FIELD, num_valid, state.stream_id)

    DS_ASSIGN(detector);
    DS_ASSIGN(track_id);

    for (auto sp : range(StepPoint::size_))
    {
        DS_ASSIGN(points[sp].time);
        DS_ASSIGN(points[sp].pos);
        DS_ASSIGN(points[sp].dir);
        DS_ASSIGN(points[sp].energy);

        copy_field(&(output->points[sp].volume_instance_ids),
                   state.scratch.points[sp].volume_instance_ids,
                   num_valid,
                   state.num_volume_levels,
                   state.stream_id);
    }

    DS_ASSIGN(event_id);
    DS_ASSIGN(parent_id);
    DS_ASSIGN(primary_id);
    DS_ASSIGN(track_step_count);
    DS_ASSIGN(step_length);
    DS_ASSIGN(weight);
    DS_ASSIGN(particle);
    DS_ASSIGN(energy_deposition);

    output->num_volume_levels = state.num_volume_levels;

#undef DS_ASSIGN

    // Copies must be complete before returning
    CELER_DEVICE_API_CALL(
        StreamSynchronize(celeritas::device().stream(state.stream_id).get()));

    CELER_ENSURE(output->detector.size() == num_valid);
    CELER_ENSURE(output->track_id.size() == num_valid);
}

//---------------------------------------------------------------------------//
/*!
 * Copy to host results from tracks that died this step.
 */
template<>
void copy_deaths<MemSpace::device>(
    DetectorStepOutput* output,
    StepStateData<Ownership::reference, MemSpace::device> const& state)
{
    CELER_EXPECT(output);

    if (state.data.death_track_id.empty())
    {
        output->deaths.clear();
        return;
    }

    ScopedProfiling profile_this{"copy-deaths"};

    auto start = device_pointer_cast(state.death_valid_id.data());
    auto end = thrust::copy_if(
        thrust_execute_on(state.stream_id),
        thrust::make_counting_iterator(0_sz),
        thrust::make_counting_iterator(state.size()),
        device_pointer_cast(state.data.death_track_id.data()),
        start,
        IsValid<TrackId>{});
    size_type const num_deaths = end - start;

    if (num_deaths == 0)
    {
        output->deaths.clear();
        return;
    }

    {
        auto execute_thread
            = detail::DeathScratchCopyExecutor{state, num_deaths};
        static KernelLauncher<decltype(execute_thread)> const launch_kernel(
            "gather-death-scratch");
        launch_kernel(num_deaths, state.stream_id, execute_thread);
    }

    output->deaths.resize(num_deaths);

    auto copy_death_field = [&](auto* dst_begin, auto const& src_col) {
        using T = std::remove_reference_t<decltype(*dst_begin)>;
        Copier<T, MemSpace::host> copy{{dst_begin, num_deaths},
                                       state.stream_id};
        copy(MemSpace::device, {src_col.data().get(), num_deaths});
    };

    std::vector<TrackId> h_track_id(num_deaths);
    std::vector<PrimaryId> h_primary_id(num_deaths);
    std::vector<ParticleId> h_particle(num_deaths);
    std::vector<Real3> h_pos(num_deaths);
    std::vector<Real3> h_dir(num_deaths);
    std::vector<TrackDeathRecord::Energy> h_energy(num_deaths);
    std::vector<real_type> h_time(num_deaths);

    copy_death_field(h_track_id.data(), state.scratch.death_track_id);
    copy_death_field(h_primary_id.data(), state.scratch.death_primary_id);
    copy_death_field(h_particle.data(), state.scratch.death_particle);
    copy_death_field(h_pos.data(), state.scratch.death_pos);
    copy_death_field(h_dir.data(), state.scratch.death_dir);
    copy_death_field(h_energy.data(), state.scratch.death_energy);
    copy_death_field(h_time.data(), state.scratch.death_time);

    CELER_DEVICE_API_CALL(
        StreamSynchronize(celeritas::device().stream(state.stream_id).get()));

    for (size_type i = 0; i < num_deaths; ++i)
    {
        output->deaths[i] = {h_track_id[i],
                             h_primary_id[i],
                             h_particle[i],
                             h_pos[i],
                             h_dir[i],
                             h_energy[i],
                             h_time[i]};
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
