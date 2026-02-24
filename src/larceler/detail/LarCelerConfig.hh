//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file larceler/detail/LarCelerConfig.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <fhiclcpp/types/Atom.h>

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//

// Top-level config for constructing a LarCelerStandalone
struct LarCelerStandaloneConfig
{
    fhicl::Atom<std::string> geometry{
        fhicl::Name{"geometry"}, fhicl::Comment{R"(GDML input filename)"}};
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
