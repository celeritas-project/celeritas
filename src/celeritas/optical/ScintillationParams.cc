//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationParams.cc
//---------------------------------------------------------------------------//
#include "ScintillationParams.hh"

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with scintillation data.
 */
ScintillationParams::ScintillationParams(ScintillationDataCRef const& data)
{
    CELER_EXPECT(data);
    HostVal<ScintillationData> host_data;

    host_data = data;
    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_ || data.spectrum.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
