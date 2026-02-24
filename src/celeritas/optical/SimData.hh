//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Shared simulation data.
 */
template<Ownership W, MemSpace M>
struct SimParamsData
{
    size_type max_steps{};
    size_type max_step_iters{};

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return max_steps > 0 && max_step_iters > 0;
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimParamsData& operator=(SimParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        max_steps = other.max_steps;
        max_step_iters = other.max_step_iters;
        return *this;
    }
};

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

    Items<PrimaryId> primary_ids;  //!< Originating primary
    Items<real_type> time;  //!< Time elapsed in lab frame since start of event
    Items<real_type> step_length;
    Items<TrackStatus> status;
    Items<ActionId> post_step_action;
    Items<size_type> num_steps;  //!< Total number of steps taken

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !primary_ids.empty() && !time.empty() && !step_length.empty()
               && !status.empty() && !post_step_action.empty()
               && !num_steps.empty();
    }

    //! State size
    CELER_FUNCTION size_type size() const { return status.size(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    SimStateData& operator=(SimStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        primary_ids = other.primary_ids;
        time = other.time;
        step_length = other.step_length;
        status = other.status;
        post_step_action = other.post_step_action;
        num_steps = other.num_steps;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Resize simulation states and set \c alive to be false.
 */
template<MemSpace M>
inline void resize(SimStateData<Ownership::value, M>* data, size_type size)
{
    CELER_EXPECT(size > 0);

    resize(&data->primary_ids, size);
    resize(&data->time, size);
    resize(&data->step_length, size);

    resize(&data->status, size);
    fill(TrackStatus::inactive, &data->status);

    resize(&data->post_step_action, size);
    resize(&data->num_steps, size);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
