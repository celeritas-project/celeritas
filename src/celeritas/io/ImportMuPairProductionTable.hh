//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportMuPairProductionTable.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "ImportPhysicsVector.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Sampling table for electron-positron pair production by muons.
 *
 * This 3-dimensional table is used to sample the energy transfer to the
 * electron-positron pair, \f$ \epsilon_p \f$. The outer grid stores the atomic
 * number using 5 equally spaced points in \f$ \log Z \f$; the x grid tabulates
 * the ratio \f$ \log \epsilon_p / T \f$, where \f$ T \f$ is the incident muon
 * energy; the y grid stores the incident muon energy using equal spacing in
 * \f$ \log T \f$.
 */
struct ImportMuPairProductionTable
{
    //!@{
    //! \name Type aliases
    using ZInt = int;
    //!@}

    std::vector<ZInt> atomic_number;
    std::vector<ImportPhysics2DVector> physics_vectors;

    explicit operator bool() const
    {
        return !atomic_number.empty()
               && physics_vectors.size() == atomic_number.size();
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
