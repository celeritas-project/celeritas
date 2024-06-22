//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ProtoInterface.cc
//---------------------------------------------------------------------------//
#include "ProtoInterface.hh"

#include <nlohmann/json.hpp>

#include "celeritas_config.h"
#include "corecel/io/JsonPimpl.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
// Get a JSON string representing a proto
std::string to_string(ProtoInterface const& proto)
{
    JsonPimpl json_wrap;
    proto.output(&json_wrap);
    return json_wrap.obj.dump();
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
