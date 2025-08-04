//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/detail/BoundaryActionTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/optical/CoreParams.hh"

#include "InitBoundaryExecutor.hh"
#include "PostBoundaryExecutor.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Type traits describing an initialization / finalization boundary action.
 *
 * Provides the action name, description, and a \c construct function to
 * create the corresponding executor.
 */
template<class E>
struct BoundaryActionTraits;

template<>
struct BoundaryActionTraits<InitBoundaryExecutor>
{
    constexpr static char const* action_name = "optical-boundary-init";
    constexpr static char const* action_desc
        = "Initialize optical boundary crossing action";

    template<MemSpace M>
    static InitBoundaryExecutor
    construct(CoreParams const& params, CoreState<M>&)
    {
        return InitBoundaryExecutor{params.surface()->ref<M>()};
    }
};

template<>
struct BoundaryActionTraits<PostBoundaryExecutor>
{
    constexpr static char const* action_name = "optical-boundary-post";
    constexpr static char const* action_desc
        = "Finalize optical boundary crossing action";

    template<MemSpace M>
    static PostBoundaryExecutor construct(CoreParams const&, CoreState<M>&)
    {
        return PostBoundaryExecutor{};
    }
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
