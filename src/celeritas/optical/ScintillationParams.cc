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
 * Construct with optical property data.
 */
ScintillationParams::ScintillationParams(OpticalPropertyCRef const& properties)
{
    HostVal<ScintillationData> data;
    for (auto mid : range(OpticalMaterialId(properties.scint_spectra.size())))
    {
        make_builder(&data.spectra).push_back(properties.scint_spectra[mid]);
    }
    mirror_ = CollectionMirror<ScintillationData>{std::move(data)};
    CELER_ENSURE(mirror_ || properties.scint_spectra.empty());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
