//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OutputInterface.cc
//---------------------------------------------------------------------------//
#include "OutputInterface.hh"

#include "celeritas_config.h"

#include "EnumStringMapper.hh"
#include "JsonPimpl.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

using Category = celeritas::OutputInterface::Category;

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Get a string corresponding to a category.
 */
char const* to_cstring(Category value)
{
    static EnumStringMapper<Category> const to_cstring_impl{
        "input",
        "result",
        "system",
        "internal",
    };
    return to_cstring_impl(value);
}

//---------------------------------------------------------------------------//
/*!
 * Get the JSON representation of a single output (mostly for testing).
 */
std::string to_string(OutputInterface const& output)
{
#if CELERITAS_USE_JSON
    JsonPimpl json_wrap;
    output.output(&json_wrap);
    return json_wrap.obj.dump();
#else
    CELER_DISCARD(output);
    return "\"output unavailable\"";
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
