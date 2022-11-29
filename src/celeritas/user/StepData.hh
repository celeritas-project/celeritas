//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
    bool volume{false};
    bool energy{false};

    //! Whether any selection is requested
    explicit CELER_FUNCTION operator bool() const
    {
        return time || pos || dir || volume || energy;
    }

    //! Combine the selection with another
    StepPointSelection& operator|=(const StepPointSelection& other)
    {
        this->time |= other.time;
        this->pos |= other.pos;
        this->dir |= other.dir;
        this->volume |= other.volume;
        this->energy |= other.energy;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Which track properties to gather at every step.
 *
 * These should correspond to the data items in StepStateData.
 */
struct StepSelection
{
    EnumArray<StepPoint, StepPointSelection> points;

    bool event{false};
    bool track_step_count{false};
    bool action{false};
    bool step_length{false};
    bool particle{false};
    bool energy_deposition{false};

    //! Whether any selection is requested
    explicit CELER_FUNCTION operator bool() const
    {
        return points[StepPoint::pre] || points[StepPoint::post] || event
               || track_step_count || action || step_length || particle
               || energy_deposition;
    }

    //! Combine the selection with another
    StepSelection& operator|=(const StepSelection& other)
    {
        for (auto sp : range(StepPoint::size_))
        {
            points[sp] |= other.points[sp];
        }

        this->event |= other.event;
        this->track_step_count |= other.track_step_count;
        this->action |= other.action;
        this->step_length |= other.step_length;
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
    Collection<DetectorId, W, M, VolumeId> detector;

    //! Filter out steps that have not deposited energy (for sensitive det)
    bool nonzero_energy_deposition{false};

    //// METHODS ////

    //! Whether the data is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return static_cast<bool>(selection);
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StepParamsData& operator=(const StepParamsData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        selection = other.selection;
        detector  = other.detector;
        nonzero_energy_deposition = other.nonzero_energy_deposition;
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
 *   evaluation) the VolumeId will be "false".
 */
template<Ownership W, MemSpace M>
struct StepPointStateData
{
    //// TYPES ////

    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    using Energy     = units::MevEnergy;

    // Sim
    StateItems<real_type> time;

    // Geo
    StateItems<Real3>    pos;
    StateItems<Real3>    dir;
    StateItems<VolumeId> volume;

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
        time   = other.time;
        pos    = other.pos;
        dir    = other.dir;
        volume = other.volume;
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
struct StepStateData
{
    //// TYPES ////

    using StepPointData = StepPointStateData<W, M>;
    template<class T>
    using StateItems = celeritas::StateCollection<T, W, M>;
    using Energy     = units::MevEnergy;

    //// DATA ////

    // Pre- and post-step data
    EnumArray<StepPoint, StepPointData> points;

    //! Track ID is always assigned (but will be false for inactive tracks)
    StateItems<TrackId> track;

    //! Detector ID is non-empty if params.detector is nonempty
    StateItems<DetectorId> detector;

    // Sim
    StateItems<EventId>   event;
    StateItems<size_type> track_step_count;
    StateItems<ActionId>  action;
    StateItems<real_type> step_length;

    // Physics
    StateItems<ParticleId> particle;
    StateItems<Energy>     energy_deposition;

    //// METHODS ////

    //! True if constructed and correctly sized
    explicit CELER_FUNCTION operator bool() const
    {
        auto right_sized = [this](const auto& t) {
            return (t.size() == this->size()) || t.empty();
        };

        return !track.empty() && right_sized(detector) && right_sized(event)
               && right_sized(track_step_count) && right_sized(action)
               && right_sized(step_length) && right_sized(particle)
               && right_sized(energy_deposition);
    }

    //! State size
    CELER_FUNCTION ThreadId::size_type size() const { return track.size(); }

    //! Assign from another set of states
    template<Ownership W2, MemSpace M2>
    StepStateData& operator=(StepStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);

        for (auto sp : range(StepPoint::size_))
        {
            points[sp] = other.points[sp];
        }

        track                   = other.track;
        detector                = other.detector;
        event                   = other.event;
        track_step_count        = other.track_step_count;
        action                  = other.action;
        step_length             = other.step_length;
        particle                = other.particle;
        energy_deposition       = other.energy_deposition;
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
                   StepPointSelection                       selection,
                   size_type                                size)
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
    SD_RESIZE_IF_SELECTED(volume);
    SD_RESIZE_IF_SELECTED(energy);

#undef SD_RESIZE_IF_SELECTED
}

//---------------------------------------------------------------------------//
/*!
 * Resize the state.
 */
template<MemSpace M>
inline void resize(StepStateData<Ownership::value, M>* state,
                   const HostCRef<StepParamsData>&     params,
                   size_type                           size)
{
    CELER_EXPECT(state->size() == 0);
    CELER_EXPECT(size > 0);

    for (auto sp : range(StepPoint::size_))
    {
        resize(&state->points[sp], params.selection.points[sp], size);
    }

#define SD_RESIZE_IF_SELECTED(ATTR)     \
    do                                  \
    {                                   \
        if (params.selection.ATTR)      \
        {                               \
            resize(&state->ATTR, size); \
        }                               \
    } while (0)

    resize(&state->track, size);
    if (!params.detector.empty())
    {
        resize(&state->detector, size);
    }

    SD_RESIZE_IF_SELECTED(event);
    SD_RESIZE_IF_SELECTED(track_step_count);
    SD_RESIZE_IF_SELECTED(step_length);
    SD_RESIZE_IF_SELECTED(action);
    SD_RESIZE_IF_SELECTED(particle);
    SD_RESIZE_IF_SELECTED(energy_deposition);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
