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
 * only construction argument is the size.
 *
 * The State class must be templated on ownership and memory space, and
 * additionally must have an operator bool(), a templated operator=, and a
 * size() accessor. It must also define a free function "resize" that takes
 * a value state and (optionally) Params host data.
 *
 * The state store is designed to be usable only from host code. Because C++
 * code in .cu files is still processed by the device compilation phase, this
 * restricts its use to .cc files currently. The embedded collection references
 * can be passed to CUDA kernels, of course. This restriction is designed to
 * reduce propagation of C++ management classes into kernel compilation to
 * improve performance of NVCC build times, and not due to a fundamental
 * capability restriction.
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

    // Construct from values by capture, mostly for testing
    explicit inline CollectionStateStore(S<Ownership::value, M>&& other);

    // Construct from values by copy, mostly for testing
    template<MemSpace M2>
    explicit inline CollectionStateStore(const S<Ownership::value, M2>& other);

    // Move construction from this memspace or another
    template<MemSpace M2>
    inline CollectionStateStore(CollectionStateStore<S, M2>&& other);

    // Move assignment from this memspace or another
    template<MemSpace M2>
    inline CollectionStateStore& operator=(CollectionStateStore<S, M2>&& other);

    //! Default copy construction/assignment
    CollectionStateStore(const CollectionStateStore&)            = default;
    CollectionStateStore& operator=(const CollectionStateStore&) = default;

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
 * Construct from values by capture, mostly for testing.
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
 * Construct from values by copy, mostly for testing.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<MemSpace M2>
CollectionStateStore<S, M>::CollectionStateStore(
    const S<Ownership::value, M2>& other)
{
    CELER_EXPECT(other);
    // Assign using const-cast because state copy operators have to be mutable
    // even when they're just copying...
    val_ = const_cast<S<Ownership::value, M2>&>(other);
    // Save reference
    ref_ = val_;
}

//---------------------------------------------------------------------------//
/*!
 * Move construction from this memspace or another.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<MemSpace M2>
CollectionStateStore<S, M>::CollectionStateStore(
    CollectionStateStore<S, M2>&& other)
    : CollectionStateStore(std::move(other.val_))
{
}

//---------------------------------------------------------------------------//
/*!
 * Move assignment from this memspace or another.
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
template<MemSpace M2>
auto CollectionStateStore<S, M>::operator=(CollectionStateStore<S, M2>&& other)
    -> CollectionStateStore<S, M>&
{
    CELER_EXPECT(other);
    // Assign
    val_ = std::move(other.val_);
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
