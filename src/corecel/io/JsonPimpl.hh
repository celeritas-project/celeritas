//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/JsonPimpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"

#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Wrapper class for exporting JSON output.
 *
 * The caller is expected to use the value directly, so replace \c obj with the
 * JSON object.
 *
 * \code
    void output(JsonPimpl* json) const
    {
#if CELERITAS_USE_JSON
        json->obj = value_;
#else
        CELER_DISCARD(json);
#endif
    }
 * \endcode
 */
struct JsonPimpl
{
#if CELERITAS_USE_JSON
    nlohmann::json obj;
#else
    JsonPimpl() = delete;
#endif
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
