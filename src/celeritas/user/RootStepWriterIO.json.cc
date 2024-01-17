//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/RootStepWriterIO.json.cc
//---------------------------------------------------------------------------//
#include "RootStepWriterIO.json.hh"

#include <string>

#include "corecel/io/JsonUtils.json.hh"

#include "RootStepWriter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, SimpleRootFilterInput& options)
{
#define SRFI_LOAD_OPTION(NAME) CELER_JSON_LOAD_OPTION(j, options, NAME)
    SRFI_LOAD_OPTION(track_id);
    SRFI_LOAD_OPTION(event_id);
    SRFI_LOAD_OPTION(parent_id);
    SRFI_LOAD_OPTION(action_id);
#undef SRFI_LOAD_OPTION
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, SimpleRootFilterInput const& options)
{
#define SRFI_SAVE_OPTION(NAME) \
    CELER_JSON_SAVE_WHEN(j, options, NAME, options.NAME != options.unspecified)

    CELER_JSON_SAVE(j, options, track_id);
    SRFI_SAVE_OPTION(event_id);
    SRFI_SAVE_OPTION(parent_id);
    SRFI_SAVE_OPTION(action_id);
#undef SRFI_SAVE_OPTION
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
