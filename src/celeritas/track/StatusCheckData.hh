//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/StatusCheckData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/Collection.hh"
#include "corecel/data/CollectionAlgorithms.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Status check parameters.
 */
template<Ownership W, MemSpace M>
struct StatusCheckParamsData
{
    //// DATA ////

    celeritas::Collection<StepActionOrder, W, M, ActionId> orders;

    static constexpr StepActionOrder implicit_order = StepActionOrder::size_;

    //// METHODS ////

    //! Whether the data are assigned
    explicit CELER_FUNCTION operator bool() const { return !orders.empty(); }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StatusCheckParamsData& operator=(StatusCheckParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        orders = other.orders;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Store the previous state and action IDs.
 */
template<Ownership W, MemSpace M>
struct StatusCheckStateData
{
    //// TYPES ////

    template<class T>
    using Items = celeritas::StateCollection<T, W, M>;

    //// DATA ////

    ActionId action;
    StepActionOrder order{StepActionOrder::size_};

    Items<TrackStatus> status;
    Items<ActionId> post_step_action;
    Items<ActionId> along_step_action;

    //// METHODS ////

    //! Check whether the interface is assigned
    explicit CELER_FUNCTION operator bool() const
    {
        return !status.empty() && !post_step_action.empty()
               && !along_step_action.empty();
    }

    //! State size
    CELER_FUNCTION TrackSlotId::size_type size() const
    {
        return status.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    StatusCheckStateData& operator=(StatusCheckStateData<W2, M2>& other)
    {
        CELER_EXPECT(other);
        action = other.action;
        order = other.order;
        status = other.status;
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
void resize(StatusCheckStateData<Ownership::value, M>* data,
            HostCRef<StatusCheckParamsData> const&,
            StreamId,
            size_type size)
{
    CELER_EXPECT(size > 0);

    resize(&data->status, size);
    fill(TrackStatus::inactive, &data->status);
    resize(&data->post_step_action, size);
    resize(&data->along_step_action, size);

    CELER_ENSURE(*data);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
