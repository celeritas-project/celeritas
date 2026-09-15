//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/DiagnosticsIO.json.cc
//---------------------------------------------------------------------------//
#include "DiagnosticsIO.json.hh"

#include "corecel/io/JsonUtils.json.hh"
#include "celeritas/user/RootStepWriterIO.json.hh"

namespace celeritas
{
namespace inp
{
//---------------------------------------------------------------------------//
//!@{
//! I/O routines for JSON

void to_json(nlohmann::json& j, ExportFiles const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, physics),
        CELER_JSON_PAIR(v, offload),
        CELER_JSON_PAIR(v, geometry),
    };
}

void from_json(nlohmann::json const& j, ExportFiles& v)
{
    CELER_JSON_LOAD_OPTION(j, v, physics);
    CELER_JSON_LOAD_OPTION(j, v, offload);
    CELER_JSON_LOAD_OPTION(j, v, geometry);
}

void to_json(nlohmann::json& j, SlotDiagnostic const& v)
{
    j = nlohmann::json{CELER_JSON_PAIR(v, basename)};
}

void from_json(nlohmann::json const& j, SlotDiagnostic& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, basename);
}

void to_json(nlohmann::json& j, Timers const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, action),
        CELER_JSON_PAIR(v, step),
    };
}

void from_json(nlohmann::json const& j, Timers& v)
{
    CELER_JSON_LOAD_OPTION(j, v, action);
    CELER_JSON_LOAD_OPTION(j, v, step);
}

void to_json(nlohmann::json& j, Counters const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, step),
        CELER_JSON_PAIR(v, event),
    };
}

void from_json(nlohmann::json const& j, Counters& v)
{
    CELER_JSON_LOAD_OPTION(j, v, step);
    CELER_JSON_LOAD_OPTION(j, v, event);
}

void to_json(nlohmann::json& j, McTruth const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, output_file),
        CELER_JSON_PAIR(v, filter),
    };
}

void from_json(nlohmann::json const& j, McTruth& v)
{
    CELER_JSON_LOAD_REQUIRED(j, v, output_file);
    CELER_JSON_LOAD_OPTION(j, v, filter);
}

void to_json(nlohmann::json& j, StepDiagnostic const& v)
{
    j = nlohmann::json{CELER_JSON_PAIR(v, bins)};
}

void from_json(nlohmann::json const& j, StepDiagnostic& v)
{
    CELER_JSON_LOAD_OPTION(j, v, bins);
}

void to_json(nlohmann::json& j, Diagnostics const& v)
{
    j = nlohmann::json{
        CELER_JSON_PAIR(v, output_file),
        CELER_JSON_PAIR(v, export_files),
        CELER_JSON_PAIR(v, timers),
        CELER_JSON_PAIR(v, counters),
        CELER_JSON_PAIR(v, perfetto_file),
        CELER_JSON_PAIR_OPTIONAL(v, slot),
        CELER_JSON_PAIR(v, action),
        CELER_JSON_PAIR(v, status_checker),
        CELER_JSON_PAIR_OPTIONAL(v, mctruth),
        CELER_JSON_PAIR_OPTIONAL(v, step),
        CELER_JSON_PAIR(v, log_frequency),
    };
}

void from_json(nlohmann::json const& j, Diagnostics& v)
{
    CELER_JSON_LOAD_OPTION(j, v, output_file);
    CELER_JSON_LOAD_OPTION(j, v, export_files);
    CELER_JSON_LOAD_OPTION(j, v, timers);
    CELER_JSON_LOAD_OPTION(j, v, counters);
    CELER_JSON_LOAD_OPTION(j, v, perfetto_file);
    CELER_JSON_LOAD_OPTIONAL(j, v, slot);
    CELER_JSON_LOAD_OPTION(j, v, action);
    CELER_JSON_LOAD_OPTION(j, v, status_checker);
    CELER_JSON_LOAD_OPTIONAL(j, v, mctruth);
    CELER_JSON_LOAD_OPTIONAL(j, v, step);
    CELER_JSON_LOAD_OPTION(j, v, log_frequency);
}

//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
