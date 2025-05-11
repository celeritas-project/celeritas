//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/StreamStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/sys/Device.hh"
#include "corecel/sys/ThreadId.hh"

#include "Collection.hh"
#include "CollectionMirror.hh"
#include "CollectionStateStore.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for storing parameters and multiple stream-dependent states.
 *
 * This requires a templated ParamsData and StateData. Hopefully this
 * frankenstein of a class will be replaced by a std::any-like data container
 * owned by each (possibly thread-local) State.
 *
 * Usage:
 * \code
   StreamStore<FooParams, FooState> store{host_val, num_streams};
   assert(store);

   execute_kernel(store.params(), store.state<Memspace::host>(StreamId{0},
 state_size))

   if (auto* state = store.state<Memspace::device>(StreamId{1}))
   {
       cout << "Have device data for stream 1" << endl;
   }
   \endcode
 *
 * There is some additional complexity in the "state" accessors to allow for
 * const correctness.
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
class StreamStore
{
  public:
    //!@{
    //! \name Type aliases
    using ParamsHostVal = P<Ownership::value, MemSpace::host>;
    //!@}

  public:
    // Default for unassigned/lazy construction
    StreamStore() = default;

    // Construct with number of streams and host data
    inline StreamStore(ParamsHostVal&& host, StreamId::size_type num_streams);

    //// ACCESSORS ////

    //! Whether the instance is ready for storing data
    explicit operator bool() const { return num_streams_ > 0; }

    //! Number of streams being stored
    StreamId::size_type num_streams() const { return num_streams_; }

    // Get references to the params data
    template<MemSpace M>
    inline P<Ownership::const_reference, M> const& params() const;

    // Get references to the state data for a given stream, allocating if
    // necessary.
    template<MemSpace M>
    inline S<Ownership::reference, M>&
    state(StreamId stream_id, size_type size);

    //! Get a pointer to the state data, null if not allocated
    template<MemSpace M>
    S<Ownership::reference, M> const* state(StreamId stream_id) const
    {
        return StreamStore::stateptr_impl<M>(*this, stream_id);
    }

    //! Get a mutable pointer to the state data, null if not allocated
    template<MemSpace M>
    S<Ownership::reference, M>* state(StreamId stream_id)
    {
        return StreamStore::stateptr_impl<M>(*this, stream_id);
    }

  private:
    //// TYPES ////
    using ParamMirror = CollectionMirror<P>;
    template<MemSpace M>
    using StateStore = CollectionStateStore<S, M>;
    template<MemSpace M>
    using VecSS = std::vector<StateStore<M>>;

    //// DATA ////

    CollectionMirror<P> params_;
    StreamId::size_type num_streams_{0};
    VecSS<MemSpace::host> host_states_;
    VecSS<MemSpace::device> device_states_;

    //// FUNCTIONS ////

    template<MemSpace M, class Self>
    static constexpr decltype(auto) states_impl(Self&& self)
    {
        if constexpr (M == MemSpace::host)
        {
            // Extra parens needed to return reference instead of copy
            return (self.host_states_);
        }
        else
        {
            return (self.device_states_);
        }
#if CELER_CUDACC_BUGGY_IF_CONSTEXPR
        CELER_ASSERT_UNREACHABLE();
#endif
    }

    template<MemSpace M, class Self>
    static decltype(auto) stateptr_impl(Self&& self, StreamId stream_id)
    {
        CELER_EXPECT(stream_id < self.num_streams_ || !self);
        using result_type = std::add_pointer_t<
            decltype(StreamStore::states_impl<M>(self).front().ref())>;
        if (!self)
        {
            return result_type{nullptr};
        }

        auto& state_vec = StreamStore::states_impl<M>(self);
        CELER_ASSERT(state_vec.size() == self.num_streams_);
        auto& state_store = state_vec[stream_id.unchecked_get()];
        if (!state_store)
        {
            return result_type{nullptr};
        }

        return &state_store.ref();
    }
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with parameters and the number of streams.
 *
 * The constructor is *not* thread safe and should be called during params
 * setup, not at run time.
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
StreamStore<P, S>::StreamStore(ParamsHostVal&& host,
                               StreamId::size_type num_streams)
    : params_(std::move(host)), num_streams_(num_streams)
{
    CELER_EXPECT(params_);
    CELER_EXPECT(num_streams_ > 0);

    // Resize stores in advance, but don't allocate memory.
    host_states_.resize(num_streams_);
    device_states_.resize(num_streams_);
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the params data.
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
template<MemSpace M>
P<Ownership::const_reference, M> const& StreamStore<P, S>::params() const
{
    CELER_EXPECT(*this);
    return params_.template ref<M>();
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the state data, allocating if necessary.
 */
template<template<Ownership, MemSpace> class P,
         template<Ownership, MemSpace> class S>
template<MemSpace M>
S<Ownership::reference, M>&
StreamStore<P, S>::state(StreamId stream_id, size_type size)
{
    CELER_EXPECT(*this);
    CELER_EXPECT(stream_id < num_streams_);

    auto& state_vec = StreamStore::states_impl<M>(*this);
    CELER_ASSERT(state_vec.size() == num_streams_);
    auto& state_store = state_vec[stream_id.unchecked_get()];
    if (CELER_UNLIKELY(!state_store))
    {
        state_store = {this->params<MemSpace::host>(), stream_id, size};
    }

    CELER_ENSURE(state_store.size() == size);
    return state_store.ref();
}

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Apply a function to all streams.
 */
template<class S, class F>
void apply_to_all_streams(S&& store, F&& func)
{
    // Apply on host
    for (StreamId s : range(StreamId{store.num_streams()}))
    {
        if (auto* state = store.template state<MemSpace::host>(s))
        {
            func(*state);
        }
    }

    // Apply on device
    for (StreamId s : range(StreamId{store.num_streams()}))
    {
        if (auto* state = store.template state<MemSpace::device>(s))
        {
            func(*state);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate data over all streams.
 */
template<class S, class F, class T>
void accumulate_over_streams(S&& store, F&& func, std::vector<T>* result)
{
    std::vector<T> temp_host;

    // Accumulate on host
    for (StreamId s : range(StreamId{store.num_streams()}))
    {
        if (auto* state = store.template state<MemSpace::host>(s))
        {
            auto data = func(*state)[AllItems<T>{}];
            CELER_EXPECT(data.size() == result->size());
            for (auto i : range(data.size()))
            {
                (*result)[i] += data[i];
            }
        }
    }

    // Accumulate on device
    for (StreamId s : range(StreamId{store.num_streams()}))
    {
        if (auto* state = store.template state<MemSpace::device>(s))
        {
            auto data = func(*state);
            CELER_EXPECT(data.size() == result->size());

            if (temp_host.empty())
            {
                temp_host.resize(result->size());
            }
            copy_to_host(data, make_span(temp_host));

            for (auto i : range(data.size()))
            {
                (*result)[i] += temp_host[i];
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
