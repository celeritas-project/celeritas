//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/DiagnosticParams.cc
//---------------------------------------------------------------------------//
#include "DiagnosticParams.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "celeritas/user/DiagnosticData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with list of enabled diagnostics.
 */
DiagnosticParams::DiagnosticParams(Input const& inp)
{
    HostVal<DiagnosticParamsData> host_data;
    host_data.field_diagnostic = inp.field_diagnostic;
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<DiagnosticParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
