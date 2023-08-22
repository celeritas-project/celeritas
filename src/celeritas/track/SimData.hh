//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/sys/ThreadId.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Particle-dependent parameters for killing looping tracks.
 *
 * These threshold values are used to determine when tracks that are flagged as
 * looping (i.e., taking a large number of substeps in the field propagator)
 * should be killed.
 *
 * In Geant4, tracks are killed immediately if their energy is below the
 * "important energy" (equivalent to \c threshold_energy here) or after some
 * number of step iterations if their energy is above the threshold.
 *
 * In Celeritas, the default \c max_substeps in the field propagator is set to
 * a smaller value than in Geant4. Therefore, an additional parameter \c
 * max_subthreshold_steps is added to approximate Geant4's policy for killing
 * looping tracks: a track flagged as looping will be killed if its energy is
 * below \c threshold_energy and it has taken more then \c
 * max_subthreshold_steps steps, or after \c max_steps steps if its energy is
 * above the threshold.
 */
struct LoopingThreshold
{
    using Energy = units::MevEnergy;

    size_type max_subthreshold_steps{10};
    size_type max_steps{100};
    Energy threshold_energy{250};

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_subthreshold_steps > 0 && max_steps > 0
               && threshold_energy >= zero_quantity();
    }
};
//---------------------------------------------------------------------------//
/*!
 * Persistent simulation data.
 */
template<Ownership W, MemSpace M>
struct SimParamsData
{
    //// TYPES ////

    template<class T>
    using ParticleItems = Collection<T, W, M, ParticleId>;

    //// DATA ////

    ParticleItems<LoopingThreshold> looping;

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const { return !looping.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimParamsData& operator=(SimParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        looping = other.looping;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Simulation state of a track.
 */
struct SimTrackInitializer
{
    TrackId track_id;  //!< Unique ID for this track
    TrackId parent_id;  //!< ID of parent that created it
    EventId event_id;  //!< ID of originating event
    real_type time{0};  //!< Time elapsed in lab frame since start of event [s]

    TrackStatus status{TrackStatus::inactive};

    //! True if assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return track_id && event_id;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Data storage/access for simulation states.
 */
template<Ownership W, MemSpace M>
struct SimStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    Items<TrackId> track_ids;  //!< Unique ID for this track
    Items<TrackId> parent_ids;  //!< ID of parent that created it
    Items<EventId> event_ids;  //!< ID of originating event
    Items<size_type> num_steps;  //!< Total number of steps taken
    Items<size_type> num_looping_steps;  //!< Number of steps taken since the
                                         //!< track was flagged as looping
    Items<real_type> time;  //!< Time elapsed in lab frame since start of event
                            //!< [s]

    Items<TrackStatus> status;
    Items<real_type> step_length;
    Items<ActionId> post_step_action;
    Items<ActionId> along_step_action;

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !track_ids.empty() && !parent_ids.empty() && !event_ids.empty()
               && !num_steps.empty() && !num_looping_steps.empty()
               && !time.empty() && !status.empty() && !step_length.empty()
               && !post_step_action.empty() && !along_step_action.empty();
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return track_ids.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimStateData& operator=(SimStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        track_ids = other.track_ids;
        parent_ids = other.parent_ids;
        event_ids = other.event_ids;
        num_steps = other.num_steps;
        num_looping_steps = other.num_looping_steps;
        time = other.time;
        status = other.status;
        step_length = other.step_length;
        post_step_action = other.post_step_action;
        along_step_action = other.along_step_action;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize simulation states and set \c alive to be false.
 */
template<MemSpace M>
void resize(SimStateData<Ownership::value, M>* data, size_type size)
{
    CELER_EXPECT(size > 0);

    resize(&data->track_ids, size);
    resize(&data->parent_ids, size);
    resize(&data->event_ids, size);

    resize(&data->num_steps, size);
    fill(size_type{0}, &data->num_steps);

    resize(&data->num_looping_steps, size);
    fill(size_type{0}, &data->num_looping_steps);

    resize(&data->time, size);
    fill(real_type{0}, &data->time);

    resize(&data->status, size);
    fill(TrackStatus::inactive, &data->status);

    resize(&data->step_length, size);
    resize(&data->post_step_action, size);
    resize(&data->along_step_action, size);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
