//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/SimData.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Storage for dynamic simulation data.
 */
template<Ownership W, MemSpace M>
struct SimStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    Items<TrackStatus> status;
    Items<ActionId> post_step_action;
    Items<ActionId> along_step_action;
    Items<real_type> step_length;
    Items<size_type> num_steps;  //!< Total number of steps taken
    Items<real_type> time;  //!< Time elapsed in lab frame since start of event

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !num_steps.empty() && !time.empty() && !status.empty()
               && !step_length.empty() && !post_step_action.empty()
               && !along_step_action.empty();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return status.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimStateData& operator=(SimStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        status = other.status;
        post_step_action = other.post_step_action;
        along_step_action = other.along_step_action;
        step_length = other.step_length;
        num_steps = other.num_steps;
        time = other.time;
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

    resize(&data->status, size);
    fill(TrackStatus::inactive, &data->status);

    resize(&data->post_step_action, size);
    resize(&data->along_step_action, size);
    resize(&data->step_length, size);
    resize(&data->num_steps, size);
    resize(&data->time, size);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
