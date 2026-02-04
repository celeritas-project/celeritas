//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/detail/LarCelerConfig.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <fhiclcpp/types/Atom.h>
#include <fhiclcpp/types/Sequence.h>
#include <fhiclcpp/types/Table.h>
#include <fhiclcpp/types/TableFragment.h>
#include <fhiclcpp/types/Tuple.h>

#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
//! FHiCL configuration table for optical states
struct OpticalStateCapacityConfig
{
    fhicl::Atom<size_type> primaries{
        fhicl::Name{"primaries"},
        fhicl::Comment{R"(Max primaries buffered before stepping)"},
    };

    fhicl::Atom<size_type> tracks{
        fhicl::Name{"tracks"},
        fhicl::Comment{R"(Max track slots stepped simultaneously)"}};

    fhicl::Atom<size_type> generators{
        fhicl::Name{"generators"},
        fhicl::Comment{R"(Max queued photon-generation steps)"}};
};

// Top-level config for constructing a LarCelerStandalone
struct LarCelerStandaloneConfig
{
#if 0
    // TODO: GPT-generated map<string,string> does not work
    fhicl::TableFragment< fhicl::Table<fhicl::Atom<std::string>> >
        environment{fhicl::Name{"environment"},
                                fhicl::Comment{R"(Environment key–value map)"}};
#endif

    fhicl::Atom<std::string> geometry{
        fhicl::Name{"geometry"}, fhicl::Comment{R"(GDML input filename)"}};

    fhicl::Atom<size_type> optical_step_iters{
        fhicl::Name{"optical_step_iters"},
        fhicl::Comment{
            R"(Iterations before aborting optical stepping loop (0 for unlimited))"},
        0};

    fhicl::Table<OpticalStateCapacityConfig> optical_capacity{
        fhicl::Name{"optical_capacity"},
        fhicl::Comment{R"(Optical buffer-size capacities)"}};

    fhicl::Atom<unsigned int> seed{fhicl::Name{"seed"},
                                   fhicl::Comment{R"(RNG seed)"}};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
