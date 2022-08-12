//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantPhysicsOptions.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Brems selection (TODO: make bitset)
enum class BremsModelSelection
{
    seltzer_berger,
    relativistic,
    all,
    size_
};

//---------------------------------------------------------------------------//
//! MSC selection (TODO: make bitset)
enum class MscModelSelection
{
    none,
    urban,
    wentzel_vi,
    size_
};

//---------------------------------------------------------------------------//
//! Construction options for geant custom physics list
struct GeantPhysicsOptions
{
    bool coulomb_scattering{false};
    bool rayleigh_scattering{true};

    BremsModelSelection brems{BremsModelSelection::all};
    MscModelSelection   msc{MscModelSelection::urban};

    int em_bins_per_decade{7};
};

//---------------------------------------------------------------------------//
} // namespace celeritas
