//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/CollectionStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Assert.hh"
#include "corecel/OpaqueId.hh"
#include "corecel/Types.hh"
#include "corecel/sys/ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class for storing Collection classes on host or device.
 *
 * This can be used for unit tests (MemSpace is host) as well as production
 * code. States generally shouldn't be copied between host and device, so the
 * only "production use case" construction argument is the size. Other
 * constructors are implemented for convenience in unit tests.
 *
 * The State class must be templated on ownership and memory space, and
 * additionally must have an operator bool(), a templated operator=, and a
 * size() accessor. It must also define a free function "resize" that takes
 * a value state and (optionally) Params host data.
 *
 * \code
    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, num_tracks);
    state_data.particle = pstates.ref();
   \endcode
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
class CollectionStateStore
{
  public:
    //!@{
    //! \name Type aliases
    using Value     = S<Ownership::value, M>;
    using Ref       = S<Ownership::reference, M>;
    using size_type = ThreadId::size_type;
    //!@}

  public:
    CollectionStateStore() = default;

    // Construct from parameters
    template<template<Ownership, MemSpace> class P>
    inline CollectionStateStore(const HostCRef<P>& p, size_type size);

    // Construct without parameters
    explicit inline CollectionStateStore(size_type size);

    // Construct from values by capture
    explicit inline CollectionStateStore(S<Ownership::value, M>&& other);

    // Copy construction from state data (convenience for unit tests)
    template<Ownership W2, MemSpace M2>
    explicit inline CollectionStateStore(const S<W2, M2>& other);

    // Copy assignment from state data (convenience for unit tests)
    template<Ownership W2, MemSpace M2>
    inline CollectionStateStore& operator=(const S<W2, M2>& other);

    //!@{
    //! Default copy/move construction/assignment
    CollectionStateStore(const CollectionStateStore&)            = default;
    CollectionStateStore& operator=(const CollectionStateStore&) = default;
    CollectionStateStore(CollectionStateStore&&)            = default;
    CollectionStateStore& operator=(CollectionStateStore&&) = default;
    //!@}

    //! Whether any data is being stored
    explicit operator bool() const { return static_cast<bool>(val_); }

    //! Number of elements
    size_type size() const { return val_.size(); }

    // Get a reference to the mutable state data
    inline const Ref& ref() const;

  private:
    Value val_;
    Ref   ref_;

    template<template<Ownership, MemSpace> class S2, MemSpace M2>
    friend class CollectionStateStore;
};

//---------------------------------------------------------------------------//
/*!
 * Construct from parameter data.
 *
 * Most states are constructed with a \c resize function that takes host
 * parameter data and the number of states.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<template<Ownership, MemSpace> class P>
CollectionStateStore<S, M>::CollectionStateStore(const HostCRef<P>& p,
                                                 size_type          size)
{
    CELER_EXPECT(size > 0);
    resize(&val_, p, size);

    // Save reference
    ref_ = val_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct without parameters.
 *
 * A few states are constructed with a \c resize function that doesn't depend
 * on any parameter data.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
CollectionStateStore<S, M>::CollectionStateStore(size_type size)
{
    CELER_EXPECT(size > 0);
    resize(&val_, size);

    // Save reference
    ref_ = val_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from values by capture.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
CollectionStateStore<S, M>::CollectionStateStore(S<Ownership::value, M>&& other)
    : val_(std::move(other))
{
    CELER_EXPECT(val_);
    // Save reference
    ref_ = val_;
}

//---------------------------------------------------------------------------//
/*!
 * Construct from a state.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<Ownership W2, MemSpace M2>
CollectionStateStore<S, M>::CollectionStateStore(const S<W2, M2>& other)
{
    CELER_EXPECT(other);
    // Assign using const-cast because state copy operators have to be mutable
    // even when they're just copying...
    val_ = const_cast<S<W2, M2>&>(other);
    // Save reference
    ref_ = val_;
}

//---------------------------------------------------------------------------//
/*!
 * Copy assign from a state.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<Ownership W2, MemSpace M2>
auto CollectionStateStore<S, M>::operator=(const S<W2, M2>& other)
    -> CollectionStateStore<S, M>&
{
    CELER_EXPECT(other);
    // Assign
    val_ = const_cast<S<W2, M2>&>(other);
    // Save reference
    ref_ = val_;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to the mutable state data.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
auto CollectionStateStore<S, M>::ref() const -> const Ref&
{
    CELER_EXPECT(*this);
    return ref_;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
