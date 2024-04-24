//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/JsonPimpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas_config.h"
#include "corecel/Assert.hh"

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
/*!
 * Helper function to write an object to JSON.
 *
 * This hides the "not configured" boilerplate.
 */
template<class T>
void to_json_pimpl(JsonPimpl* jp, T const& self)
{
#if CELERITAS_USE_JSON
    CELER_EXPECT(jp);
    to_json(jp->obj, self);
#else
    CELER_NOT_CONFIGURED("JSON");
#endif
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
