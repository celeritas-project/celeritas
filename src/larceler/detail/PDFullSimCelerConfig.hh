//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/detail/PDFullSimCelerConfig.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <canvas/Utilities/InputTag.h>
#include <fhiclcpp/types/Atom.h>

#include "corecel/Types.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
// Top-level config for constructing a PDFullSimCeler
struct PDFullSimCelerConfig
{
    fhicl::Atom<art::InputTag> SimulationLabel{
        fhicl::Name{"SimulationLabel"},
        fhicl::Comment{R"(SimEnergyDeposit event tag)"}};

    // GPU options
    fhicl::Atom<bool> EnableDevice{fhicl::Name{"EnableDevice"},
                                   fhicl::Comment{R"(Activate the GPU)"},
                                   false};

    fhicl::Atom<size_type> DeviceStackSize{
        fhicl::Name{"DeviceStackSize"},
        fhicl::Comment{R"(Per-thread CUDA stack size [B] (ignored if 0))"},
        0};

    fhicl::Atom<size_type> DeviceHeapSize{
        fhicl::Name{"DeviceHeapSize"},
        fhicl::Comment{R"(Global dynamic CUDA heap size [B] (ignored if 0))"},
        0};

    // Optical buffer-size capacities
    fhicl::Atom<size_type> OpticalCapacityPrimaries{
        fhicl::Name{"OpticalCapacityPrimaries"},
        fhicl::Comment{R"(Max primaries buffered before stepping)"}};

    fhicl::Atom<size_type> OpticalCapacityTracks{
        fhicl::Name{"OpticalCapacityTracks"},
        fhicl::Comment{R"(Max track slots stepped simultaneously)"}};

    fhicl::Atom<size_type> OpticalCapacityGenerators{
        fhicl::Name{"OpticalCapacityGenerators"},
        fhicl::Comment{R"(Max queued photon-generation steps)"}};

    // Optical tracking limits
    fhicl::Atom<size_type> OpticalLimitSteps{
        fhicl::Name{"OpticalLimitSteps"},
        fhicl::Comment{R"(Steps per track before killing (0 for unlimited))"},
        0};

    fhicl::Atom<size_type> OpticalLimitStepIters{
        fhicl::Name{"OpticalLimitStepIters"},
        fhicl::Comment{
            R"(Iterations before aborting stepping loop (0 for unlimited))"},
        0};

    fhicl::Atom<std::string> OutputFile{
        fhicl::Name{"OutputFile"},
        fhicl::Comment{R"(Celeritas output filename)"}};

    fhicl::Atom<bool> ActionTimes{
        fhicl::Name{"ActionTimes"},
        fhicl::Comment{R"(Accumulate elapsed time in actions)"},
        false};

    fhicl::Atom<unsigned int> Seed{
        fhicl::Name{"Seed"}, fhicl::Comment{R"(RNG seed)"}, 12345};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
