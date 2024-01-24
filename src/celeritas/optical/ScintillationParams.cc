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
 * Construct with scintillation input data.
 */
ScintillationParams::ScintillationParams(ScintillationInput const& input)
{
    HostVal<ScintillationData> host_data;

    CollectionBuilder spectra(&host_data.spectra);
    CollectionBuilder components(&host_data.components);
    for (auto const& spec : input.data)
    {
        ScintillationSpectrum spectrum;
        spectrum.components = components.insert_back(spec.begin(), spec.end());
        spectra.push_back(spectrum);
    }

    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
