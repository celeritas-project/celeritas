//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldDriverOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "FieldDriverOptionsIO.json.hh"

#include <string>
#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"

#include "FieldDriverOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, FieldDriverOptions& opts)
{
#define FDO_INPUT(NAME)                    \
    do                                     \
    {                                      \
        if (j.contains(#NAME))             \
            j.at(#NAME).get_to(opts.NAME); \
    } while (0)

    FDO_INPUT(minimum_step);
    FDO_INPUT(delta_chord);
    FDO_INPUT(delta_intersection);
    FDO_INPUT(epsilon_step);
    FDO_INPUT(epsilon_rel_max);
    FDO_INPUT(errcon);
    FDO_INPUT(pgrow);
    FDO_INPUT(pshrink);
    FDO_INPUT(safety);
    FDO_INPUT(max_stepping_increase);
    FDO_INPUT(max_stepping_decrease);
    FDO_INPUT(max_nsteps);
    FDO_INPUT(max_substeps);

#undef FDO_INPUT
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, FieldDriverOptions const& opts)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(opts, minimum_step),
        CELER_JSON_PAIR(opts, delta_chord),
        CELER_JSON_PAIR(opts, delta_intersection),
        CELER_JSON_PAIR(opts, epsilon_step),
        CELER_JSON_PAIR(opts, epsilon_rel_max),
        CELER_JSON_PAIR(opts, errcon),
        CELER_JSON_PAIR(opts, pgrow),
        CELER_JSON_PAIR(opts, pshrink),
        CELER_JSON_PAIR(opts, safety),
        CELER_JSON_PAIR(opts, max_stepping_increase),
        CELER_JSON_PAIR(opts, max_stepping_decrease),
        CELER_JSON_PAIR(opts, max_nsteps),
        CELER_JSON_PAIR(opts, max_substeps),
    };
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
