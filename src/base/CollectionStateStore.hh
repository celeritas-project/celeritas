//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CollectionStateStore.hh
//---------------------------------------------------------------------------//
#pragma once

#include "Assert.hh"
#include "OpaqueId.hh"
#include "Types.hh"

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
 * a value state and (optionally) Params host pointers.
 *
 * The state store is designed to be usable only from host code. Because C++
 * code in .cu files is still processed by the device compilation phase, this
 * restricts its use to .cc files currently. The embededd collection references
 * can be passed to CUDA kernels, of course. This restriction is designed to
 * reduce propagation of C++ management classes into kernel compilation to
 * improve performance of NVCC build times, and not due to a fundamental
 * capability restriction.
 *
 * \code
    CollectionStateStore<ParticleStateData, MemSpace::device> pstates(
        *particle_params, num_tracks);
    state_pointers.particle = pstates.ref();
   \endcode
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
class CollectionStateStore
{
  public:
    //!@{
    //! Type aliases
    using Value     = S<Ownership::value, M>;
    using Ref       = S<Ownership::reference, M>;
    using size_type = ThreadId::size_type;
    //!@}

  public:
    CollectionStateStore() = default;

    //! Construct from parameters
    template<class Params>
    CollectionStateStore(const Params& p, size_type size)
    {
#ifdef __CUDA_ARCH__
        static_assert(sizeof(Params) == 0,
                      "Collection state store is not designed for CUDA device "
                      "compilation phase");
#endif
        CELER_EXPECT(size > 0);
        resize(&val_, p.host_pointers(), size);

        // Save reference
        ref_ = val_;
    }

    //! Construct without parameters
    explicit CollectionStateStore(size_type size)
    {
        CELER_EXPECT(size > 0);
        resize(&val_, size);

        // Save reference
        ref_ = val_;
    }

    //! Whether any data is being stored
    explicit operator bool() const { return static_cast<bool>(val_); }

    //! Number of elements
    size_type size() const { return val_.size(); }

    //! Get a reference to the mutable state data
    const Ref& ref() const
    {
        CELER_EXPECT(*this);
        return ref_;
    }

  private:
    Value val_;
    Ref   ref_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas
