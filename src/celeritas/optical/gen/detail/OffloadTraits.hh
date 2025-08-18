//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/OffloadTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
class CherenkovParams;
class ScintillationParams;

namespace detail
{
struct CherenkovOffloadExecutor;
struct ScintOffloadExecutor;

//---------------------------------------------------------------------------//
//! Traits for offloading optical distribution data from a process
template<GeneratorType G>
struct OffloadTraits;

template<>
struct OffloadTraits<GeneratorType::cherenkov>
{
    //! Params class
    using Params = CherenkovParams;

    //! Optical offload executor
    using Executor = CherenkovOffloadExecutor;

    //! Label of the offload action
    static constexpr char const label[] = "cherenkov-offload";

    //! Description of the offload action
    static constexpr char const description[]
        = "generate Cherenkov optical distribution data";
};

template<>
struct OffloadTraits<GeneratorType::scintillation>
{
    //! Params class
    using Params = ScintillationParams;

    //! Optical offload executor
    using Executor = ScintOffloadExecutor;

    //! Label of the offload action
    static constexpr char const label[] = "scintillation-offload";

    //! Description of the offload action
    static constexpr char const description[]
        = "generate scintillation optical distribution data";
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
