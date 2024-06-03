//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/InputBuilder.cc
//---------------------------------------------------------------------------//
#include "InputBuilder.hh"

#include <fstream>

#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeLog.hh"
#include "corecel/sys/ScopedMem.hh"
#include "corecel/sys/ScopedProfiling.hh"

#include "ProtoInterface.hh"

#include "detail/ProtoBuilder.hh"
#include "detail/ProtoMap.hh"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>

#    include "corecel/io/JsonPimpl.hh"
#endif

namespace celeritas
{
namespace orangeinp
{
namespace
{
//---------------------------------------------------------------------------//
void write_protos(detail::ProtoMap const& map, std::string const& filename)
{
    if (!CELERITAS_USE_JSON)
    {
        CELER_LOG(warning)
            << "JSON support is not enabled: no proto output written to \""
            << filename << '"';
        CELER_DISCARD(map);
    }

#if CELERITAS_USE_JSON
    auto result = nlohmann::json(std::vector<nullptr_t>(map.size()));
    for (auto uid : range(UniverseId{map.size()}))
    {
        JsonPimpl j;
        map.at(uid)->output(&j);
        result[uid.get()] = std::move(j.obj);
    }

    std::ofstream outf(filename);
    CELER_VALIDATE(outf,
                   << "failed to open output file at \"" << filename << '"');
    outf << result.dump();

    CELER_LOG(info) << "Wrote ORANGE protos to " << filename;
#endif
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with options.
 */
InputBuilder::InputBuilder(Options&& opts) : opts_{std::move(opts)}
{
    CELER_EXPECT(opts_.tol);
}

//---------------------------------------------------------------------------//
/*!
 * Construct an ORANGE geometry.
 */
auto InputBuilder::operator()(ProtoInterface const& global) const -> result_type
{
    ScopedProfiling profile_this{"build-orange-input"};
    ScopedMem record_mem("orange.build_input");
    CELER_LOG(status) << "Constructing ORANGE surfaces and runtime data";
    ScopedTimeLog scoped_time;

    // Construct the hierarchy of protos
    detail::ProtoMap const protos{global};
    CELER_ASSERT(protos.find(&global) == orange_global_universe);
    if (!opts_.proto_output_file.empty())
    {
        write_protos(protos, opts_.proto_output_file);
    }

    // Build surfaces and metadata
    OrangeInput result;
    detail::ProtoBuilder builder(&result, opts_.tol, protos);
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
