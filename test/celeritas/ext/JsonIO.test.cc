//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/JsonIO.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantOpticalPhysicsOptions.hh"
#include "celeritas/ext/GeantOpticalPhysicsOptionsIO.json.hh"

#include "JsonUtils.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
TEST(JsonIOTest, optical)
{
    auto input = GeantOpticalPhysicsOptions::deactivated();
    input.absorption = true;

    char const expected[]
        = R"json({"_format":"geant4-optical-physics","_version":"0.7.0","absorption":true,"boundary":null,"cherenkov":null,"mie_scattering":false,"rayleigh_scattering":false,"scintillation":null,"verbose":false,"wavelength_shifting":null,"wavelength_shifting2":null})json";
    verify_json_round_trip(input, expected);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
