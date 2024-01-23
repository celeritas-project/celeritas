//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/AssertIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "Assert.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
void to_json(nlohmann::json&, DebugErrorDetails const&);
void to_json(nlohmann::json&, RuntimeErrorDetails const&);
void to_json(nlohmann::json&, RichContextException const&);

//---------------------------------------------------------------------------//
}  // namespace celeritas
