//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/detail/GeneratorTraits.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Types.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
template<Ownership W, MemSpace M>
struct CherenkovData;
class CherenkovParams;
template<Ownership W, MemSpace M>
struct ScintillationData;
class ScintillationParams;

namespace optical
{
class CherenkovGenerator;
class ScintillationGenerator;

namespace detail
{
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
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
