//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/detail/JsonUtils.json.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

//---------------------------------------------------------------------------//
// MACROS
//---------------------------------------------------------------------------//
/*!
 * Load a required field into a struct.
 */
#define CELER_JSON_LOAD_REQUIRED(OBJ, STRUCT, NAME) \
    OBJ.at(#NAME).get_to(STRUCT.NAME)

/*!
 * Load an optional field.
 *
 * If the field is missing or null, it is omitted.
 */
#define CELER_JSON_LOAD_OPTION(OBJ, STRUCT, NAME)  \
    do                                             \
    {                                              \
        if (auto iter = OBJ.find(#NAME);           \
            iter != OBJ.end() && !iter->is_null()) \
        {                                          \
            iter->get_to(STRUCT.NAME);             \
        }                                          \
    } while (0)

/*!
 * Load an optional field.
 *
 * If the field is missing or null, it is omitted.
 */
#define CELER_JSON_LOAD_DEPRECATED(OBJ, STRUCT, OLD, NEW)         \
    do                                                            \
    {                                                             \
        if (auto iter = OBJ.find(#OLD); iter != OBJ.end())        \
        {                                                         \
            ::celeritas::warn_deprecated_json_option(#OLD, #NEW); \
            iter->get_to(STRUCT.NEW);                             \
        }                                                         \
    } while (0)

/*!
 * Save a field to a JSON object.
 */
#define CELER_JSON_SAVE(OBJ, STRUCT, NAME) OBJ[#NAME] = STRUCT.NAME

/*!
 * Save a field if the condition is met.
 *
 * If not met, a "null" placeholder is saved.
 */
#define CELER_JSON_SAVE_WHEN(OBJ, STRUCT, NAME, COND) \
    do                                                \
    {                                                 \
        if ((COND))                                   \
        {                                             \
            CELER_JSON_SAVE(OBJ, STRUCT, NAME);       \
        }                                             \
        else                                          \
        {                                             \
            OBJ[#NAME] = nullptr;                     \
        }                                             \
    } while (0)
//---------------------------------------------------------------------------//

namespace celeritas
{
//---------------------------------------------------------------------------//
// Print a warning about a deprecated input option
void warn_deprecated_json_option(char const* old_name, char const* new_name);

// Save a format and version marker
void save_format(nlohmann::json& j, std::string const& format);

// Load and check for a format and compatible version marker
void check_format(nlohmann::json const& j, std::string const& format);

//---------------------------------------------------------------------------//
}  // namespace celeritas
