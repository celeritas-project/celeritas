//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/ExceptionOutput.cc
//---------------------------------------------------------------------------//
#include "ExceptionOutput.hh"

#include "corecel/sys/TypeDemangler.hh"
#include "celeritas_config.h"
#include "JsonPimpl.hh"
#if CELERITAS_USE_JSON
#    include <nlohmann/json.hpp>
#endif


namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with an exception object.
 *
 * This saves the type information and message for later output.
 */
ExceptionOutput::ExceptionOutput(const std::exception& e)
{
    TypeDemangler<std::exception> demangle;
    type_ = demangle(e);
    what_ = e.what();
}

//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void ExceptionOutput::output(JsonPimpl* j) const
{
#if CELERITAS_USE_JSON
    j->obj = {
        {"type", type_},
        {"what", what_}
    };
#else
    (void)sizeof(j);
#endif
}

//---------------------------------------------------------------------------//
} // namespace celeritas
