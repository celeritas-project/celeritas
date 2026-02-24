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

namespace celeritas
{
namespace detail
{

// Top-level config for constructing a LarCelerStandalone
struct LarCelerStandaloneConfig
{
    fhicl::Atom<std::string> output_file{
        fhicl::Name{"output_file"},
        fhicl::Comment{R"(Celeritas output filename)"}};

    fhicl::Atom<bool> action_times{
        fhicl::Name{"action_times"},
        fhicl::Comment{R"(Accumulate elapsed time in actions)"},
        false};

    fhicl::Atom<unsigned int> seed{
        fhicl::Name{"seed"}, fhicl::Comment{R"(RNG seed)"}, 0};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
