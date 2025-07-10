//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"

namespace celeritas
{
template<Ownership W, MemSpace M>
struct CherenkovData;
class CherenkovGenerator;
class CherenkovParams;
template<Ownership W, MemSpace M>
struct ScintillationData;
class ScintillationGenerator;
class ScintillationParams;

namespace detail
{
struct CherenkovOffloadExecutor;
struct ScintOffloadExecutor;

//---------------------------------------------------------------------------//
//! Process used to generate optical photons
enum class GeneratorType
{
    cherenkov,
    scintillation,
};

//---------------------------------------------------------------------------//
//! Traits for generating optical photons from a process
template<GeneratorType G>
struct GeneratorTraits;

template<>
struct GeneratorTraits<GeneratorType::cherenkov>
{
    //! Shared process data
    template<Ownership W, MemSpace M>
    using Data = CherenkovData<W, M>;

    //! Params class
    using Params = CherenkovParams;

    //! Optical photon generator
    using Generator = CherenkovGenerator;

    //! Label of the generator action
    static constexpr char const label[] = "cherenkov-generate";

    //! Description of the generator action
    static constexpr char const description[]
        = "generate Cherenkov photons from optical distribution data";
};

template<>
struct GeneratorTraits<GeneratorType::scintillation>
{
    //! Shared process data
    template<Ownership W, MemSpace M>
    using Data = ScintillationData<W, M>;

    //! Params class
    using Params = ScintillationParams;

    //! Optical photon generator
    using Generator = ScintillationGenerator;

    //! Label of the generator action
    static constexpr char const label[] = "scintillation-generate";

    //! Description of the generator action
    static constexpr char const description[]
        = "generate scintillation photons from optical distribution data";
};

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
