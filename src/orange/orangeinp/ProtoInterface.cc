//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ProtoInterface.cc
//---------------------------------------------------------------------------//
#include "ProtoInterface.hh"

#include "celeritas_config.h"
#include "corecel/io/JsonPimpl.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
// Get a JSON string representing a proto
std::string to_string(ProtoInterface const& proto)
{
#if CELERITAS_USE_JSON
    JsonPimpl json_wrap;
    proto.output(&json_wrap);
    return json_wrap.obj.dump();
#else
    CELER_DISCARD(proto);
    return "\"output unavailable\"";
#endif
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
