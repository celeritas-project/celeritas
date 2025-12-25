//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/JsonPimpl.hh
//---------------------------------------------------------------------------//
#pragma once

#include <nlohmann/json.hpp>

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

//! Get a JSON object from an OutputInterface
template<class T>
nlohmann::json output_to_json(T const& self)
{
    JsonPimpl jp;
    self.output(&jp);
    return std::move(jp.obj);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
