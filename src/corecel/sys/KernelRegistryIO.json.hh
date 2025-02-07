//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/KernelRegistryIO.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "KernelRegistry.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//

// Write one kernel's attributes to json
void to_json(nlohmann::json& j, KernelAttributes const& md);
// Write all kernels to JSON
void to_json(nlohmann::json& j, KernelRegistry const& diagnostics);

//---------------------------------------------------------------------------//
}  // namespace celeritas
