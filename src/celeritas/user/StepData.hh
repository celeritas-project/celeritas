//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/StepData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// TYPES
//---------------------------------------------------------------------------//
/*!
 * Which track properties to gather at the beginning and end of a step.
 *
 * These should all default to *false* so that this list can be extended
 * without adversely affecting existing interfaces.
 */
struct StepPointSelection
{
    bool time{false};
    bool pos{false};
    bool dir{false};
    bool energy{false};

    bool volume_id{false};
    bool volume_instance_ids{false};

    //! Create StepPointSelection with all options set to true
    static constexpr StepPointSelection all()
    {
        return StepPointSelection{true, true, true, true, true, true};
    }

    //! Whether any selection is requested
    explicit CELER_FUNCTION operator bool() const
    {
        return time || pos || dir || energy || volume_id || volume_instance_ids;
    }

    //! Combine the selection with another
    StepPointSelection& operator|=(StepPointSelection const& other)
    {
        this->time |= other.time;
        this->pos |= other.pos;
        this->dir |= other.dir;
        this->energy |= other.energy;
        this->volume_id |= other.volume_id;
        this->volume_instance_ids |= other.volume_instance_ids;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Which track properties to gather at every step.
 *
 * These should correspond to the data items in StepStateDataImpl.
 *
 * TODO: particle -> particle_id for consistency
 */
struct StepSelection
{
    EnumArray<StepPoint, StepPointSelection> points;

    bool event_id{false};
    bool parent_id{false};
    bool primary_id{false};
    bool track_step_count{false};
    bool action_id{false};
    bool step_length{false};
    bool weight{false};
    bool particle{false};
    bool energy_deposition{false};

    //! Create StepSelection with all options set to true
    static constexpr StepSelection all()
    {
        return StepSelection{
            {StepPointSelection::all(), StepPointSelection::all()},
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true};
    }

    //! Whether any selection is requested
    explicit CELER_FUNCTION operator bool() const
    {
        return points[StepPoint::pre] || points[StepPoint::post] || event_id
               || parent_id || track_step_count || action_id || step_length
               || weight || particle || energy_deposition;
    }

    //! Combine the selection with another
    StepSelection& operator|=(StepSelection const& other)
    {
        for (auto sp : range(StepPoint::size_))
        {
            points[sp] |= other.points[sp];
        }

        this->event_id |= other.event_id;
        this->parent_id |= other.parent_id;
        this->track_step_count |= other.track_step_count;
        this->action_id |= other.action_id;
        this->step_length |= other.step_length;
        this->weight |= other.weight;
        this->particle |= other.particle;
        this->energy_deposition |= other.energy_deposition;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Shared attributes about the hits being collected.
 *
 * This will be expanded to include filters for particle type, region, etc.
 */
template<Ownership W, MemSpace M>
struct StepParamsData
{
    //// DATA ////

    //! Options for gathering data at each step
    StepSelection selection;

    //! Optional mapping for volume -> sensitive detector
    Collection<DetectorId, W, M, ImplVolumeId> detector;

    //! Filter out steps that have not deposited energy (for sensitive det)
    bool nonzero_energy_deposition{false};

    //! Per-state volume instance size if volume_instance_ids selected
    size_type volume_instance_depth{0};

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(selection);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StepParamsData& operator=(StepParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        selection = other.selection;
        detector = other.detector;
        nonzero_energy_deposition = other.nonzero_energy_deposition;
        volume_instance_depth = other.volume_instance_depth;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Gathered state data for beginning/end of step data for tracks in parallel.
 *
 * - Each data member corresponds exactly to a flag in \c StepPointSelection
 * - If the flag is disabled (no step interfaces require the data), then the
 *   corresponding member data will be empty.
 * - If a track is outside the volume (which can only happen at the end-of-step
 *   evaluation) the ImplVolumeId will be "false".
 */
template<Ownership W, MemSpace M>
struct StepPointStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    template<class T>
    using Items = celeritas::Collection<T, W, M>;
    using Energy = units::MevEnergy;

    // Sim
    StateItems<real_type> time;

    // Geo
    StateItems<Real3> pos;
    StateItems<Real3> dir;
    StateItems<ImplVolumeId> volume_id;
    Items<VolumeInstanceId> volume_instance_ids;

    // Physics
    StateItems<Energy> energy;

    //// METHODS ////

    //! Always true since all step-point data could be disabled
    explicit CELER_FUNCTION operator bool() const { return true; }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    StepPointStateData& operator=(StepPointStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        time = other.time;
        pos = other.pos;
        dir = other.dir;
        volume_id = other.volume_id;
        volume_instance_ids = other.volume_instance_ids;
        energy = other.energy;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Gathered data for a single step for many tracks in parallel.
 *
 * - Each data member corresponds exactly to a flag in \c StepSelection .
 * - If the flag is disabled (no step interfaces require the data), then the
 *   corresponding member data will be empty.
 * - The track ID will be set to "false" if the track is inactive.
 * - If sensitive detector are specified, the \c detector field is set based
 *   on the pre-step geometric volume. Data members will have \b unspecified
 *   values if the detector ID is "false" (i.e. no information is being
 *   collected). The detector ID for inactive threads is always "false".
 */
template<Ownership W, MemSpace M>
struct StepStateDataImpl
{
    //// TYPES ////

    using StepPointData = StepPointStateData<W, M>;
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    using Energy = units::MevEnergy;

    //// DATA ////

    // Pre- and post-step data
    EnumArray<StepPoint, StepPointData> points;

    //! Track ID is always assigned (but will be false for inactive tracks)
    StateItems<TrackId> track_id;

    //! Detector ID is non-empty if params.detector is nonempty
    StateItems<DetectorId> detector;

    // Sim
    StateItems<EventId> event_id;
    StateItems<TrackId> parent_id;
    StateItems<PrimaryId> primary_id;
    StateItems<ActionId> action_id;
    StateItems<size_type> track_step_count;
    StateItems<real_type> step_length;
    StateItems<real_type> weight;

    // Physics
    StateItems<ParticleId> particle;
    StateItems<Energy> energy_deposition;

    //// METHODS ////

    //! True if constructed and correctly sized
    explicit CELER_FUNCTION operator bool() const
    {
        auto right_sized = [this](auto const& t) {
            return (t.size() == this->size()) || t.empty();
        };

        return !track_id.empty() && right_sized(detector)
               && right_sized(event_id) && right_sized(parent_id)
               && right_sized(primary_id) && right_sized(track_step_count)
               && right_sized(action_id) && right_sized(step_length)
               && right_sized(weight) && right_sized(particle)
               && right_sized(energy_deposition);
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return track_id.size();
    }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    StepStateDataImpl& operator=(StepStateDataImpl<W2, M2>& other)
    {
        // The extra storage used to gather the step data is only required on
        // the device
        CELER_EXPECT(other || (M == MemSpace::host && other.size() == 0));

        for (auto sp : range(StepPoint::size_))
        {
            points[sp] = other.points[sp];
        }

        track_id = other.track_id;
        parent_id = other.parent_id;
        primary_id = other.primary_id;
        detector = other.detector;
        event_id = other.event_id;
        track_step_count = other.track_step_count;
        action_id = other.action_id;
        step_length = other.step_length;
        weight = other.weight;
        particle = other.particle;
        energy_deposition = other.energy_deposition;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Gathered data and persistent scratch space for gathering and copying data.
 *
 * Extra storage \c scratch and \c valid_id is needed to efficiently gather and
 * copy the step data on the device but will not be allocated on the host.
 *
 * \todo Refactor the step interface stuff so that we passing params alongside
 * state data to the step interfaces, so that we don't have to keep a copy of
 * the volume instance depth (and for better consistency with the rest of the
 * code).
 */
template<Ownership W, MemSpace M>
struct StepStateData
{
    //// TYPES ////

    using StepDataImpl = StepStateDataImpl<W, M>;
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    //! Gathered data for a single step
    StepDataImpl data;

    //! Scratch space for gathering the data on device based on track validity
    StepDataImpl scratch;

    //! Thread IDs of active tracks that are in a detector
    StateItems<size_type> valid_id;

    // Copy of params max depth for dimensioning volume_instance_ids
    size_type volume_instance_depth{0};

    //! Unique identifier for "thread-local" data.
    StreamId stream_id;

    //// METHODS ////

    //! True if constructed and correctly sized
    explicit CELER_FUNCTION operator bool() const
    {
        auto right_sized = [this](auto const& t) {
            return (t.size() == this->size())
                   || (t.size() == 0 && M == MemSpace::host);
        };

        return data.size() > 0 && right_sized(scratch) && right_sized(valid_id)
               && stream_id;
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const { return data.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    StepStateData& operator=(StepStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        data = other.data;
        scratch = other.scratch;
        valid_id = other.valid_id;
        volume_instance_depth = other.volume_instance_depth;
        stream_id = other.stream_id;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Resize a state point.
 */
template<MemSpace M>
inline void resize(StepPointStateData<Ownership::value, M>* state,
                   StepPointSelection selection,
                   HostCRef<StepParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(size > 0);
#define SD_RESIZE_IF_SELECTED(ATTR)     \
    do                                  \
    {                                   \
        if (selection.ATTR)             \
        {                               \
            resize(&state->ATTR, size); \
        }                               \
    } while (0)

    SD_RESIZE_IF_SELECTED(time);
    SD_RESIZE_IF_SELECTED(pos);
    SD_RESIZE_IF_SELECTED(dir);
    SD_RESIZE_IF_SELECTED(volume_id);
    if (selection.volume_instance_ids)
    {
        size_type const vi_depth = params.volume_instance_depth;
        CELER_ASSERT(vi_depth > 0);
        resize(&state->volume_instance_ids, size * vi_depth);
    }
    SD_RESIZE_IF_SELECTED(energy);

#undef SD_RESIZE_IF_SELECTED
}

//---------------------------------------------------------------------------//
/*!
 * Resize the step data.
 */
template<MemSpace M>
inline void resize(StepStateDataImpl<Ownership::value, M>* state,
                   HostCRef<StepParamsData> const& params,
                   size_type size)
{
    CELER_EXPECT(state->size() == 0);
    CELER_EXPECT(size > 0);

    for (auto sp : range(StepPoint::size_))
    {
        resize(&state->points[sp], params.selection.points[sp], params, size);
    }

#define SD_RESIZE_IF_SELECTED(ATTR)     \
    do                                  \
    {                                   \
        if (params.selection.ATTR)      \
        {                               \
            resize(&state->ATTR, size); \
        }                               \
    } while (0)

    resize(&state->track_id, size);
    if (!params.detector.empty())
    {
        resize(&state->detector, size);
    }

    SD_RESIZE_IF_SELECTED(event_id);
    SD_RESIZE_IF_SELECTED(parent_id);
    SD_RESIZE_IF_SELECTED(track_step_count);
    SD_RESIZE_IF_SELECTED(step_length);
    SD_RESIZE_IF_SELECTED(weight);
    SD_RESIZE_IF_SELECTED(action_id);
    SD_RESIZE_IF_SELECTED(particle);
    SD_RESIZE_IF_SELECTED(energy_deposition);
}

//---------------------------------------------------------------------------//
/*!
 * Resize the state.
 */
template<MemSpace M>
inline void resize(StepStateData<Ownership::value, M>* state,
                   HostCRef<StepParamsData> const& params,
                   StreamId stream_id,
                   size_type size)
{
    CELER_EXPECT(state->size() == 0);
    CELER_EXPECT(size > 0);

    state->volume_instance_depth = params.volume_instance_depth;
    state->stream_id = stream_id;

    resize(&state->data, params, size);

    if constexpr (M == MemSpace::device)
    {
        // Allocate extra space on device for gathering step data
        resize(&state->scratch, params, size);
        resize(&state->valid_id, size);
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
