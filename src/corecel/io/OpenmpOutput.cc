//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/OpenmpOutput.cc
//---------------------------------------------------------------------------//
#include "OpenmpOutput.hh"

#include <nlohmann/json.hpp>

#include "corecel/sys/Openmp.hh"

#include "JsonPimpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write output to the given JSON object.
 */
void OpenmpOutput::output(JsonPimpl* j) const
{
    j->obj = nlohmann::json::object({
        {"max_threads", openmp_max_threads()},
        {"thread_limit", openmp_thread_limit()},
        {"proc_bind", openmp_proc_bind()},
    });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
