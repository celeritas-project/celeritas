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

//! FHiCL configuration table for optical limits
struct OpticalTrackingLimitsConfig
{
    fhicl::Atom<size_type> steps{
        fhicl::Name{"steps"},
        fhicl::Comment{R"(Steps per track before killing (0 for unlimited))"},
        0};

    fhicl::Atom<size_type> step_iters{
        fhicl::Name{"step_iters"},
        fhicl::Comment{
            R"(Iterations before aborting stepping loop (0 for unlimited))"},
        0};
};

//! FHiCL configuration table for GPU capabilities and debugging options
struct DeviceConfig
{
    fhicl::Atom<bool> enable{
        fhicl::Name{"enable"}, fhicl::Comment{R"(Activate the GPU)"}, false};

    fhicl::Atom<size_type> stack_size{
        fhicl::Name{"stack_size"},
        fhicl::Comment{R"(Per-thread CUDA stack size [B] (ignored if 0))"},
        0};

    fhicl::Atom<size_type> heap_size{
        fhicl::Name{"heap_size"},
        fhicl::Comment{R"(Global dynamic CUDA heap size [B] (ignored if 0))"},
        0};
};

// Top-level config for constructing a LarCelerStandalone
struct LarCelerStandaloneConfig
{
#if 0
    // TODO: GPT-generated map<string,string> does not work
    fhicl::TableFragment< fhicl::Table<fhicl::Atom<std::string>> >
        environment{fhicl::Name{"environment"},
                                fhicl::Comment{R"(Environment key-value map)"}};
#endif

    fhicl::Table<DeviceConfig> device{fhicl::Name{"device"},
                                      fhicl::Comment{R"(GPU options)"},
                                      DeviceConfig{}};

    fhicl::Atom<std::string> geometry{
        fhicl::Name{"geometry"}, fhicl::Comment{R"(GDML input filename)"}};

    fhicl::Table<OpticalStateCapacityConfig> optical_capacity{
        fhicl::Name{"optical_capacity"},
        fhicl::Comment{R"(Optical buffer-size capacities)"}};

    fhicl::Table<OpticalTrackingLimitsConfig> optical_limits{
        fhicl::Name{"optical_limits"},
        fhicl::Comment{R"(Optical tracking limits)"},
        OpticalTrackingLimitsConfig{}};

    fhicl::Atom<std::string> output_file{
        fhicl::Name{"output_file"},
        fhicl::Comment{R"(Celeritas output filename)"}};

    fhicl::Atom<bool> action_times{
        fhicl::Name{"action_times"},
        fhicl::Comment{R"(Accumulate elapsed time in actions)"},
        false};

    fhicl::Atom<unsigned int> seed{
        fhicl::Name{"seed"}, fhicl::Comment{R"(RNG seed)"}, 12345};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
