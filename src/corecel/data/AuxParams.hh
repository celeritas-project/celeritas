//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "AuxState.hh"
#include "AuxStateVec.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct and manage portable dynamic data.
 * \tparam P Params collection group
 * \tparam S State collection group
 *
 * This generalization of \rstref{the Celeritas data model,api_data_model}
 * manages some of the boilerplate code for the common use case of having
 * portable "params" data (e.g., model data) and "state" data (e.g., temporary
 * values used across multiple kernels or processed into user space). Each
 * state/stream will have an instance of \c AuxState accessible by this
 * class. An instance of this class can be shared among multiple actions, or an
 * action could inherit from it. 
 *
 * \par Example:
 * The \c StepParams inherits from this class to provide access to host and
 * state data. The execution inside \c StepGatherAction provides views to both
 * the params and state data classes:
 * \code
    // Extract the local step state data
    auto const& step_params = params_->ref<MemSpace::native>();
    auto& step_state = params_->ref<MemSpace::native>(state.aux());

    // Run the action
    auto execute = TrackExecutor{
        params.ptr<MemSpace::native>(),
        state.ptr(),
        detail::StepGatherExecutor<P>{step_params, step_state}};
   \endcode
 *
 * \note For the case where the aux state data contains host-side classes and
 * data (e.g., an open file handle) you must manually set up the params/state
 * data using \c AuxStateInterface and \c AuxParamsInterface .
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
class AuxParams : public AuxParamsInterface, public ParamsDataInterface<P>
{
  public:
    //!@{
    //! \name Type aliases
    template<MemSpace M>
    using StateRefT = S<Ownership::reference, M>;
    //!@}

  public:
    // Factory function for building multithread state for a stream
    inline UPState create_state(MemSpace, StreamId, size_type size) const final;

    // Access the *state* ref (const)
    template<MemSpace M>
    inline StateRefT<M> const& state_ref(AuxStateVec const& v) const;

    // Access the *state* ref (mutable)
    template<MemSpace M>
    inline StateRefT<M>& state_ref(AuxStateVec& v) const;

    //! Allow access host/device *params* ref
    using ParamsDataInterface<P>::ref;

  private:
    template<MemSpace M>
    using StateT = AuxState<S, M>;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Build a multithread state for a stream.
 *
 * See \c AuxState .
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
auto AuxParams<P, S>::create_state(MemSpace m,
                                   StreamId sid,
                                   size_type size) const -> UPState
{
    return ::celeritas::make_aux_state<S, P>(*this, m, sid, size);
}

//---------------------------------------------------------------------------//
/*!
 * Access the *state* ref (const).
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
template<MemSpace M>
auto AuxParams<P, S>::state_ref(AuxStateVec const& v) const
    -> StateRefT<M> const&
{
    return ::celeritas::get<StateT<M>>(v, this->aux_id()).ref();
}

//---------------------------------------------------------------------------//
/*!
 * Access the *state* ref (mutable).
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
template<MemSpace M>
auto AuxParams<P, S>::state_ref(AuxStateVec& v) const -> StateRefT<M>&
{
    return ::celeritas::get<StateT<M>>(v, this->aux_id()).ref();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
