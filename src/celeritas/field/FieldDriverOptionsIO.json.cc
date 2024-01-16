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

#undef FDO_INPUT
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, FieldDriverOptions const& opts)
{
#define FDO_PAIR(FIELD) {#FIELD, opts.FIELD}
    j = nlohmann::json{
        FDO_PAIR(minimum_step),
        FDO_PAIR(delta_chord),
        FDO_PAIR(delta_intersection),
        FDO_PAIR(epsilon_step),
        FDO_PAIR(epsilon_rel_max),
        FDO_PAIR(errcon),
        FDO_PAIR(pgrow),
        FDO_PAIR(pshrink),
        FDO_PAIR(safety),
        FDO_PAIR(max_stepping_increase),
        FDO_PAIR(max_stepping_decrease),
        FDO_PAIR(max_nsteps),
    };
#undef FDO_PAIR
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
