//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/JsonPimpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

#include "celeritas_config.h"
#include "corecel/Assert.hh"

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
        json->obj = value_;
    }
 * \endcode
 */
struct JsonPimpl
{
    nlohmann::json obj;
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
    CELER_EXPECT(jp);
    to_json(jp->obj, self);
}

template<class T>
nlohmann::json json_pimpl_output(T const& self)
{
    JsonPimpl jp;
    self.output(&jp);
    return std::move(jp.obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
