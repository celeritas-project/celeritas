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
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "detail/InputBuilder.hh"
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
    return "\"output unavailable\"";
#endif
}

//---------------------------------------------------------------------------//
/*!
 * Construct all universes.
 */
OrangeInput build_input(Tolerance<> const& tol, ProtoInterface const& global)
{
    ScopedProfiling profile_this{"build-orange-geo"};
    ScopedMem record_mem("orangeinp::build_input");
    ScopedTimeLog scoped_time;

    OrangeInput result;
    detail::ProtoMap const protos{global};
    CELER_ASSERT(protos.find(&global) == orange_global_universe);
    detail::InputBuilder builder(&result, tol, protos);
    for (auto uid : range(UniverseId{protos.size()}))
    {
        protos.at(uid)->build(builder);
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
