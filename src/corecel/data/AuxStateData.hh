//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxStateData.hh
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
 * Helper class for retrieving templated state data on a single stream.
 *
 * This class is most easily used with \c make_aux_state to create a
 * "collection group"-style state (see \ref collections) associated with a
 * \c AuxParamsInterface subclass.
 *
 * The state class \c S must have a \c resize method that's constructable with
 * a templated params data class \c P, a stream ID, and a state size.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
class AuxStateData final : public AuxStateInterface
{
  public:
    //!@{
    //! \name Type aliases
    using Ref = S<Ownership::reference, M>;
    //!@}

  public:
    // Construct by resizing and passing host params
    template<template<Ownership, MemSpace> class P>
    inline AuxStateData(HostCRef<P> const& p,
                        StreamId stream_id,
                        size_type size);

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

//---------------------------------------------------------------------------//
/*!
 * Create an auxiliary state given a runtime memory space.
 *
 * Example:
 * \code
    return make_aux_state<ParticleTallyStateData>(
        *this, memspace, stream, size);
 * \endcode
 */
template<template<Ownership, MemSpace> class S, template<Ownership, MemSpace> class P>
std::unique_ptr<AuxStateInterface>
make_aux_state(ParamsDataInterface<P> const& params,
               MemSpace m,
               StreamId stream_id,
               size_type size)
{
    if (m == MemSpace::host)
    {
        using ASD = AuxStateData<S, MemSpace::host>;
        return std::make_unique<ASD>(params.host_ref(), stream_id, size);
    }
    else if (m == MemSpace::device)
    {
        using ASD = AuxStateData<S, MemSpace::device>;
        return std::make_unique<ASD>(params.host_ref(), stream_id, size);
    }
    CELER_ASSERT_UNREACHABLE();
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct by resizing and passing host params.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<template<Ownership, MemSpace> class P>
AuxStateData<S, M>::AuxStateData(HostCRef<P> const& p,
                                 StreamId stream_id,
                                 size_type size)
    : store_{p, stream_id, size}
{
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
