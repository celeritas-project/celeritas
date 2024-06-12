//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/neutron/model/CascadeOptionsIO.json.cc
//---------------------------------------------------------------------------//
#include "CascadeOptionsIO.json.hh"

#include <string>
#include <nlohmann/json.hpp>

#include "corecel/io/JsonUtils.json.hh"

#include "CascadeOptions.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read options from JSON.
 */
void from_json(nlohmann::json const& j, CascadeOptions& opts)
{
#define FDO_INPUT(NAME)                    \
    do                                     \
    {                                      \
        if (j.contains(#NAME))             \
            j.at(#NAME).get_to(opts.NAME); \
    } while (0)

    FDO_INPUT(use_precompound);
    FDO_INPUT(use_abla);
    FDO_INPUT(use_three_body_mom);
    FDO_INPUT(use_phase_space);
    FDO_INPUT(do_coalescence);
    FDO_INPUT(prob_pion_absorption);
    FDO_INPUT(use_two_params);
    FDO_INPUT(radius_scale);
    FDO_INPUT(radius_small);
    FDO_INPUT(radius_alpha);
    FDO_INPUT(radius_trailing);
    FDO_INPUT(fermi_scale);
    FDO_INPUT(xsec_scale);
    FDO_INPUT(gamma_qd_scale);
    FDO_INPUT(dp_max_doublet);
    FDO_INPUT(dp_max_triplet);
    FDO_INPUT(dp_max_alpha);

#undef FDO_INPUT
}

//---------------------------------------------------------------------------//
/*!
 * Write options to JSON.
 */
void to_json(nlohmann::json& j, CascadeOptions const& opts)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(opts, use_precompound),
        CELER_JSON_PAIR(opts, use_abla),
        CELER_JSON_PAIR(opts, use_three_body_mom),
        CELER_JSON_PAIR(opts, use_phase_space),
        CELER_JSON_PAIR(opts, do_coalescence),
        CELER_JSON_PAIR(opts, prob_pion_absorption),
        CELER_JSON_PAIR(opts, use_two_params),
        CELER_JSON_PAIR(opts, radius_scale),
        CELER_JSON_PAIR(opts, radius_small),
        CELER_JSON_PAIR(opts, radius_alpha),
        CELER_JSON_PAIR(opts, radius_trailing),
        CELER_JSON_PAIR(opts, fermi_scale),
        CELER_JSON_PAIR(opts, xsec_scale),
        CELER_JSON_PAIR(opts, gamma_qd_scale),
        CELER_JSON_PAIR(opts, dp_max_doublet),
        CELER_JSON_PAIR(opts, dp_max_triplet),
        CELER_JSON_PAIR(opts, dp_max_alpha),
    };
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
