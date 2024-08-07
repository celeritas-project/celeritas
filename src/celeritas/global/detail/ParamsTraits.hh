//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/detail/ParamsTraits.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
template<MemSpace M>
class CoreState;
class ExplicitCoreActionInterface;

// TODO: Add optical params and state

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Map Params class to the corresponding state and explicit action class
 */
template<typename T>
struct ParamsTraits;

template<>
struct ParamsTraits<CoreParams>
{
    template<MemSpace M>
    using State = CoreState<M>;

    using ExplicitAction = ExplicitCoreActionInterface;
};

// TODO: add explicit template instantiation of ParamsTraits for optical data

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
