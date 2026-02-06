//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Ref.hh
//! \brief Helper functions for memspace-specific references
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct a reference object pointing to state data.
 *
 * Since the "reference" type is a value whose scope must extend beyond all
 * references to it, it's often necessary to create a "reference" instance from
 * a "value" instance. Collection groups don't define templated copy
 * constructors, so this function (and the others like it) provide a
 * workaround.
 *
 * \code
   auto my_states = make_ref(my_state_values);
 * \endcode
 */
template<template<Ownership, MemSpace> class S, MemSpace M>
inline S<Ownership::reference, M> make_ref(S<Ownership::value, M>& states)
{
    S<Ownership::reference, M> result;
    result = states;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a reference object pointing to params data.
 */
template<template<Ownership, MemSpace> class P, MemSpace M>
inline P<Ownership::const_reference, M>
make_ref(P<Ownership::value, M> const& params)
{
    P<Ownership::const_reference, M> result;
    result = params;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Construct a const reference object pointing to params data.
 */
template<template<Ownership, MemSpace> class P, MemSpace M>
inline decltype(auto) make_const_ref(P<Ownership::value, M> const& params)
{
    return make_ref(params);
}

//---------------------------------------------------------------------------//
/*!
 * Get a reference to memory-spaced data owned by a params/state object.
 *
 * The object must have \c host_ref and \c device_ref accessors depending on
 * the value of \c M.
 */
template<MemSpace M, class T>
inline decltype(auto) get_ref(T&& obj)
{
    if constexpr (M == MemSpace::host)
    {
        return std::forward<T>(obj).host_ref();
    }
    else if constexpr (M == MemSpace::device)
    {
        return std::forward<T>(obj).device_ref();
    }
    else
    {
        CELER_ASSERT_UNREACHABLE();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Copy an entire collection group to the host.
 * \tparam CG CollectionGroup
 *
 * This is mostly useful for debugging and testing. It is \em not performant
 * and should not be used as part of the stepping loop, since it is likely to
 * perform many allocations.
 *
 * \code
   auto my_states = make_host_val(states.device_ref());
 * \endcode
 */
template<template<Ownership, MemSpace> class CG, Ownership W, MemSpace M>
inline auto make_host_val(CG<W, M> const& source)
{
    CG<Ownership::value, MemSpace::host> result;
    // Assign from mutable value since the "state" collections require it
    result = const_cast<CG<W, M>&>(source);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
