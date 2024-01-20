//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Ref.hh
//! \brief Helper functions for memspace-specific references
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

#include "detail/RefImpl.hh"

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
decltype(auto) get_ref(T&& obj)
{
    return detail::RefGetter<T, M>{std::forward<T>(obj)}();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
