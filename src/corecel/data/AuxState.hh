//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxState.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Types.hh"

#include "AuxInterface.hh"
#include "CollectionStateStore.hh"
#include "ParamsDataInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Store a state collection group as aux state data.
 * \tparam S State data collection group
 * \tparam M Memory space for the state
 *
 * This class is most easily used with \c make_aux_state to create a
 * "collection group"-style state (see \ref collections) associated with a
 * \c AuxParamsInterface subclass.
 *
 * The state class \c S must have a \c resize method that's constructable with
 * an optional templated params data class \c P, a stream ID, and a state size.
 *
 * The \c make_aux_state helper functions can be used to construct this class:
 * \code
    return make_aux_state<FooStateData>(*this, memspace, stream, size);
 * \endcode
 * or
 * \code
    return make_aux_state<BarStateData>(memspace, stream, size);
 * \endcode
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
class AuxState final : public AuxStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Ref = S<Ownership::reference, M>;
    //!@}

  public:
    // Construct by resizing and passing host params
    template<template<Ownership, MemSpace> class P>
    inline AuxState(HostCRef<P> const& p, StreamId stream_id, size_type size);

    // Construct by resizing without params
    inline AuxState(StreamId stream_id, size_type size);

    //! Whether any data is being stored
    explicit operator bool() const { return static_cast<bool>(store_); }

    //! Number of elements in the state
    size_type size() const { return store_.size(); }

    //! Get a reference to the mutable state data
    Ref& ref() { return store_.ref(); }

    //! Get a reference to immutable state data
    Ref const& ref() const { return store_.ref(); }

  private:
    CollectionStateStore<S, M> store_;
};

//! \cond (CELERITAS_DOC_DEV)
//---------------------------------------------------------------------------//
/*!
 * Create an auxiliary state given a runtime memory space and host params.
 */
template<template<Ownership, MemSpace> class S,
         template<Ownership, MemSpace> class P>
std::unique_ptr<AuxStateInterface>
make_aux_state(ParamsDataInterface<P> const& params,
               MemSpace m,
               StreamId stream_id,
               size_type size)
{
    if (m == MemSpace::host)
    {
        using ASD = AuxState<S, MemSpace::host>;
        return std::make_unique<ASD>(params.host_ref(), stream_id, size);
    }
    else if (m == MemSpace::device)
    {
        using ASD = AuxState<S, MemSpace::device>;
        return std::make_unique<ASD>(params.host_ref(), stream_id, size);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
/*!
 * Create an auxiliary state given a runtime memory space.
 */
template<template<Ownership, MemSpace> class S>
std::unique_ptr<AuxStateInterface>
make_aux_state(MemSpace m, StreamId stream_id, size_type size)
{
    if (m == MemSpace::host)
    {
        using ASD = AuxState<S, MemSpace::host>;
        return std::make_unique<ASD>(stream_id, size);
    }
    else if (m == MemSpace::device)
    {
        using ASD = AuxState<S, MemSpace::device>;
        return std::make_unique<ASD>(stream_id, size);
    }
    CELER_ASSERT_UNREACHABLE();
}
//! \endcond

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct by resizing and passing host params.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<template<Ownership, MemSpace> class P>
AuxState<S, M>::AuxState(HostCRef<P> const& p,
                         StreamId stream_id,
                         size_type size)
    : store_{p, stream_id, size}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct by resizing.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
AuxState<S, M>::AuxState(StreamId stream_id, size_type size)
    : store_{stream_id, size}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
