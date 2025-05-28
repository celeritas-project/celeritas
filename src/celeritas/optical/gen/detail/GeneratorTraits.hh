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
struct CherenkovGenerator;
template<Ownership W, MemSpace M>
struct ScintillationData;
struct ScintillationGenerator;

namespace detail
{
//---------------------------------------------------------------------------//
//! Process used to generate optical photons
enum class GeneratorType
{
    cherenkov,
    scintillation,
};

//---------------------------------------------------------------------------//
template<GeneratorType G>
struct GeneratorTraits;

template<>
struct GeneratorTraits<GeneratorType::cherenkov>
{
    //! Shared process data
    template<Ownership W, MemSpace M>
    using Data = CherenkovData<W, M>;

    //! Optical photon generator
    using Generator = CherenkovGenerator;

    //! Label of the generating action
    static constexpr char const label[] = "generate-cherenkov-photons";

    //! Description of the generating action
    static constexpr char const description[]
        = "generate Cherenkov photons from optical distribution data";
};

template<>
struct GeneratorTraits<GeneratorType::scintillation>
{
    //! Shared process data
    template<Ownership W, MemSpace M>
    using Data = ScintillationData<W, M>;

    //! Optical photon generator
    using Generator = ScintillationGenerator;

    //! Label of the generating action
    static constexpr char const label[] = "generate-scintillation-photons";

    //! Description of the generating action
    static constexpr char const description[]
        = "generate scintillation photons from optical distribution data";
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
