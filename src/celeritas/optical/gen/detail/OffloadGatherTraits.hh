//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadGatherTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/global/ActionInterface.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace detail
{
struct OffloadPreGatherExecutor;
struct OffloadPrePostGatherExecutor;

//---------------------------------------------------------------------------//
//! Traits for gathering state data needed for optical distributions
template<StepActionOrder S>
struct OffloadGatherTraits;

template<>
struct OffloadGatherTraits<StepActionOrder::pre>
{
    //! Pre-step gather data
    template<Ownership W, MemSpace M>
    using Data = OffloadPreStateData<W, M>;

    //! Optical gather executor
    using Executor = OffloadPreGatherExecutor;

    //! Label of the gather action
    static constexpr char const label[] = "offload-pre-gather";

    //! Description of the gather action
    static constexpr char const description[]
        = "gather pre-step data for optical distributions";
};

template<>
struct OffloadGatherTraits<StepActionOrder::pre_post>
{
    //! Pre-post-step gather data
    template<Ownership W, MemSpace M>
    using Data = OffloadPrePostStateData<W, M>;

    //! Optical gather executor
    using Executor = OffloadPrePostGatherExecutor;

    //! Label of the gather action
    static constexpr char const label[] = "offload-pre-post-gather";

    //! Description of the gather action
    static constexpr char const description[]
        = "gather pre-post-step data for scintillation optical distributions";
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
