//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/RootStepWriterIO.json.cc
//---------------------------------------------------------------------------//
#include "RootStepWriterIO.json.hh"

#include <string>

#include "RootStepWriter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, SimpleRootFilterInput& options)
{
#define SRFI_LOAD_OPTION(NAME)                \
    do                                        \
    {                                         \
        if (j.contains(#NAME))                \
            j.at(#NAME).get_to(options.NAME); \
    } while (0)
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
    j["track_id"] = options.track_id;
#define SRFI_SAVE_OPTION(NAME)                   \
    do                                           \
    {                                            \
        if (options.NAME != options.unspecified) \
            j[#NAME] = options.NAME;             \
    } while (0)
    SRFI_SAVE_OPTION(event_id);
    SRFI_SAVE_OPTION(parent_id);
    SRFI_SAVE_OPTION(action_id);
#undef SRFI_SAVE_OPTION
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
